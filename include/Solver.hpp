#ifndef MPC_POMDP_SOLVER_HEADER_FILE
#define MPC_POMDP_SOLVER_HEADER_FILE

#include <iostream>
#include <nlopt.hpp>

#include "Model.hpp"
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

	class POMDPSolver {
		public:

            struct OptimizerData {
                Belief* belief;
                const Model & model;
                double epsilon;
                int horizon;
            };

            struct EqConData {
                const Model & model;
                int horizon;
                int N;
            };

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

			POMDPSolver(int h, double e);

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

            void setEpsilon(double e);

            double getEpsilon() const;

            /**
             * @brief This function solves a MPC_POMDP::Model completely.
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
             *
             * @return 
             */
            void operator()(const Model & model, const size_t init_state, Belief& belief);

        private:
        	size_t S, A, O;
        	int horizon_;
        	double epsilon_;

        	mutable RandomEngine rand_;

        	static double cost(const std::vector<double> &gamma, std::vector<double> &grad, void* fdata);

        	static double ineq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, void* cdata);

            static double eq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, void* cdata);
	};
}

#endif