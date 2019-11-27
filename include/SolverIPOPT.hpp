#ifndef MPC_POMDP_SOLVER_IPOPT_HEADER_FILE
#define MPC_POMDP_SOLVER_IPOPT_HEADER_FILE

#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <nlopt.hpp>

#include "models/SparseModel.hpp"
#include "models/DenseModel.hpp"
#include "POMDP_NLP.hpp"
#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "utilities/Seeder.hpp"
#include "utilities/Utils.hpp"

namespace MPC_POMDP {
    /**
     * @brief This class implements the MPC POMDP algorithm.
     *
     * The algorithm will assume a probabilistic distribution of selecting control
     * at step t. Given a chance constraints rate, this solver tries to solve at 
     * each time step the best probabilistic distribution of the control.
	 * 
     * The solver utilizes the nlopt library at each time step to solve a nonlinear
     * constrained optimization problem to maximize the rewards under current
     * belief.
     */

	class POMDPSolver_IPOPT {
		public:
			/**
             * @brief Basic constructor.
             *
             * This constructor sets the default horizon used to solve a MPC_POMDP::Model.
             *
             * The parameter h is the length of the horizon we are looking at in MPC.
             * h must >= 1, otherwise the constructor will throw an std::runtime_error.
             *
             * @param h The horizon chosen.
             * @param e The epsilon chosen. (Chance constraints parameter.)
             */

			POMDPSolver_IPOPT(int h, double e);

            /**
             * @brief This function allows setting the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
			void setHorizon(int h);

            /**
             * @brief This function returns the currently set horizon parameter.
             *
             * @return The current horizon.
             */
            int getHorizon() const;

            /**
             * @brief This function allows setting the epsilon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setEpsilon(double e);

            /**
             * @brief This function returns the currently set epsilon parameter.
             *
             * @return The current epsilon.
             */
            double getEpsilon() const;

            /**
             * @brief This function solves a MPC_POMDP::Model or 
             * MPC_POMDP::SparseModel completely.
             *
             * This function is pretty expensive (as are possibly all POMDP
             * solvers). For each step of the horizon, it computes the 
             * probabilistic distribution of the control. We pick the
             * control distribution of the first step and use that to propagate
             * the system.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             * @param init_state The initial state to start propagation.
             * @param belief Dense vector of the initial state probability pdistribution.
             */
            // template<typename M, typename B>
            void operator()(const SparseModel & model, const size_t init_state, Belief & belief);

        private:
        	size_t S, A, O;
        	int horizon_;
        	double epsilon_;

        	mutable RandomEngine rand_;
	};

    void POMDPSolver_IPOPT::operator()(const SparseModel & model, const size_t init_state, Belief & belief) {
        S = model.getS();
        A = model.getA();
        O = model.getO();

        std::ofstream ofs;
        ofs.open("results.POMDP", std::ofstream::out | std::ofstream::trunc);

        int timestep = 0;

        // Create a new instance of your nlp
        //  (use a SmartPtr, not raw)
        SmartPtr<TNLP> mynlp = new POMDP_NLP(model, belief, horizon_, epsilon_);

        // Create a new instance of IpoptApplication
        //  (use a SmartPtr, not raw)
        // We are using the factory, since this allows us to compile this
        // example with an Ipopt Windows DLL
        SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

        // Change some options
        // Note: The following choices are only examples, they might not be
        //       suitable for your optimization problem.
        app->Options()->SetNumericValue("tol", 1e-7);
        app->Options()->SetStringValue("mu_strategy", "adaptive");
        app->Options()->SetStringValue("output_file", "ipopt.out");
        app->Options()->SetStringValue("hessian_approximation", "limited-memory");
        app->Options()->SetStringValue("linear_solver", "ma27");
        // The following overwrites the default name (ipopt.opt) of the options file
        // app->Options()->SetStringValue("option_file_name", "hs071.opt");

        size_t curr_state = init_state;

        double tot_cost = 0;

        while(!model.isTermination(curr_state)) {
            std::vector<double> gamma(A*horizon_, 1.0/(double)A);
            double cost_temp = 0;

            clock_t t = clock();

            // Initialize the IpoptApplication and process the options
            ApplicationReturnStatus status;
            status = app->Initialize();
            if( status != Solve_Succeeded )
            {
                std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
                std::cout << (int) status << std::endl;
            }

            // Ask Ipopt to solve the problem
            status = app->OptimizeTNLP(mynlp);

            if( status == Solve_Succeeded )
            {
                std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
            }
            else
            {
                std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
            }

            // As the SmartPtrs go out of scope, the reference count
            // will be decremented and the objects will automatically
            // be deleted.

            std::cout << (int) status << std::endl;

            t = clock() - t;

            std::cout << "Step: " << timestep << std::endl;
            std::cout << "Computational time for this step is : " << (double) t/CLOCKS_PER_SEC << std::endl;

            assert(false);

            // std::cout << "Cost: " << cost_temp << std::endl;
            ofs << "Step: " << timestep << std::endl;
            ofs << "Cost: " << cost_temp << std::endl;
            for (size_t i = 0; i < horizon_*A; ++i) {
                ofs << gamma[i] << " ";
            }
            ofs << std::endl;
            // std::cout << "Number of evaluations: " << count << std::endl;

            tot_cost += cost_temp;

            size_t action = sampleProbability(A, gamma, rand_);

            std::tuple<size_t, size_t, double> SOR = model.propagateSOR(curr_state, action);

            Belief belief_temp(S);
            updateBelief(model, belief, action, std::get<1>(SOR), &belief_temp);
            belief = belief_temp;

            ofs << "A: " << action << " S: " << std::get<0>(SOR) << " O: " << std::get<1>(SOR) << std::endl;
            
            ofs << "Belief: " << std::endl;
            for(size_t i = 0; i < S; i ++) {
                if(checkDifferentSmall(belief(i), 0.0))
                    ofs << "S: " << i << " B: " << belief(i) << std::endl;
            }
            ofs << std::endl;

            curr_state = std::get<0>(SOR);

            ++timestep;
        }

        ofs.close();
        return;
    }

    POMDPSolver_IPOPT::POMDPSolver_IPOPT(const int h, const double e) :
    rand_(Seeder::getSeed())
    {
        setHorizon(h);
        setEpsilon(e);
    }

    void POMDPSolver_IPOPT::setHorizon(const int h) {
        if ( h < 1 ) throw std::invalid_argument("Horizon must be >= 1");
        horizon_ = h;
    }

    void POMDPSolver_IPOPT::setEpsilon(const double e) {
        if(e > 1 || e < 0) throw std::invalid_argument("Epsilon must be >= 0 and <= 1");
        epsilon_ = e;
    }

    int POMDPSolver_IPOPT::getHorizon() const {
        return horizon_;
    }
}

#endif