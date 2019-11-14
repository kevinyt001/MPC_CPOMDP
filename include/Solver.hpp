#ifndef MPC_POMDP_SOLVER_HEADER_FILE
#define MPC_POMDP_SOLVER_HEADER_FILE

#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
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
            template<typename M>
            void operator()(const M & model, const size_t init_state, Belief& belief);

        private:
        	size_t S, A, O;
        	int horizon_;
        	double epsilon_;

        	mutable RandomEngine rand_;

            template<typename M>
        	static double cost(const std::vector<double> &gamma, std::vector<double> &grad, void* fdata);

            template<typename M>
        	static double ineq_con(const std::vector<double> &gamma, std::vector<double> &grad, void* cdata);

            template<typename M>
            static double eq_con(const std::vector<double> &gamma, std::vector<double> &grad, void* cdata);

            template<typename M>
            static void eq_con_vec(unsigned int m, double* result, unsigned int n, const double* gamma, double* grad, void* cdata);

            template<typename M>
            static void ineq_con_vec(unsigned int m, double* result, unsigned int n, const double* gamma, double* grad, void* cdata);

            template<typename M>
            static void ineq_con_vec_2(unsigned int m, double* result, unsigned int n, const double* gamma, double* grad, void* cdata);

            template<typename M>
            struct OptimizerData {
                const M & model;
                const Belief & belief;
                double epsilon;
                int horizon;
            };

            template<typename M>
            struct EqConData {
                const M & model;
                int horizon;
                int N;
            };
	};

    template<typename M>
    void POMDPSolver::operator()(const M & model, const size_t init_state, Belief& belief) {
        S = model.getS();
        A = model.getA();
        O = model.getO();

        std::ofstream ofs;
        ofs.open("results.POMDP", std::ofstream::out | std::ofstream::trunc);

        int timestep = 0;

        size_t curr_state = init_state;

        OptimizerData<M> OD = {model, belief, epsilon_, horizon_};

        nlopt::opt opt(nlopt::LD_SLSQP, A*horizon_);
        opt.set_min_objective(cost<M>, &OD);
        opt.add_inequality_constraint(ineq_con<M>, &OD, 0.0);

        // Set lower bound and upper bound on gamma
        opt.set_lower_bounds(0.0);
        opt.set_upper_bounds(1.0);      

        /*
        // Add Equlaity Constraints One by One
        std::vector<EqConData> eqcon_data(horizon_, {model, horizon_, 0});
        for (int i = 0; i < horizon_; i++) {
          eqcon_data[i].N = i;
          opt.add_equality_constraint(eq_constraint, &eqcon_data[i], equalToleranceGeneral);
         }
        */

        // Add Vector Valued Equality Constraints
        EqConData<M> eqcon_data({model, horizon_, 0});
        std::vector<double> tol(horizon_, equalToleranceSmall);
        // opt.add_equality_mconstraint(eq_constraint_vector<M>, &eqcon_data, tol);
        opt.add_inequality_mconstraint(ineq_con_vec<M>, &eqcon_data, tol);
        opt.add_inequality_mconstraint(ineq_con_vec_2<M>, &eqcon_data, tol);

        // Set termination
        opt.set_ftol_abs(0.0001);
        // opt.set_xtol_abs(0.0000001);
        // opt.set_stopval(192.7);

        double tot_cost = 0;

        while(!model.isTermination(curr_state)) {
            std::vector<double> gamma(A*horizon_, 1.0/(double)A);
            double cost_temp = 0;

            clock_t t = clock();

            std::cout << "Optimization Starts" << std::endl;

            try{
                // nlopt::result result = opt.optimize(gamma, cost_temp);
                opt.optimize(gamma, cost_temp);
            }
            catch(std::exception &e) {
                std::cout << "nlopt failed: " << e.what() << std::endl;
            }

            t = clock() - t;

            std::cout << "Step: " << timestep << std::endl;
            std::cout << "Computational time for this step is : " << (double) t/CLOCKS_PER_SEC << std::endl;

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

    template<typename M>
    double POMDPSolver::cost(const std::vector<double> &gamma, std::vector<double> &grad, 
        void* fdata) {
        if (!grad.empty()) {
            std::fill(grad.begin(), grad.end(), 0);
            std::vector<double> temp_gamma = gamma;
            std::vector<double> temp;
            for(size_t i = 0; i < gamma.size(); ++i) {
                temp_gamma[i] = std::min(gamma[i] + std::max(gamma[i]*0.05, 0.01), 1.0);
                double adj1 = temp_gamma[i] - gamma[i];
                double temp1 = cost<M>(temp_gamma, temp, fdata);
                temp_gamma[i] = std::max(gamma[i] - std::max(gamma[i]*0.05, 0.01), 0.0);
                double adj2 = temp_gamma[i] - gamma[i];
                double temp2 = cost<M>(temp_gamma, temp, fdata);

                grad[i] = (temp1 - temp2) / (adj1 - adj2);

                temp_gamma[i] = gamma[i];
            }
        }

        OptimizerData<M> *bm = reinterpret_cast<OptimizerData<M>*>(fdata);
        const M & m = bm->model; const Belief & b = bm->belief; int h = bm->horizon;

        double cost = 0;
        Belief predict_belief = b;
        // clock_t t = clock();
        for (int i = 0; i < h; ++i) {
            // Eigen::Map<Vector> g(gamma.data()+i*h, m.getA());
            Vector g(m.getA());
            for (size_t j = 0; j < m.getA(); ++j) 
                g(j) = gamma[i*h+j];

            /*
            // rewards_.transpose: A*S; belief: S*1
            Vector temp_grad = m.getRewardFunction().transpose() * predict_belief;
            // temp_grad.transpose: 1*A; g(gamma): A*1
            cost += temp_grad.transpose() * g;

            if (!grad.empty()) {
                for(size_t j = 0; j < m.getA(); ++j)
                    grad[i*h+j] = temp_grad(j);
            }
            */

            // belief.transpose: 1*S; rewards_: S*A; g(gamma): A*1
            cost += predict_belief.transpose() * m.getRewardFunction() * g;

            // Update belief
            if(i == h) break;
            Belief temp = predict_belief;

            //Update together
            predict_belief.setZero();
            for(size_t j = 0; j < m.getA(); ++j) {
                predict_belief.noalias() += Eigen::VectorXd(g(j) * m.getTransitionFunction(j).transpose() * temp);
            }
            // std::cout << predict_belief << std::endl;

            //Update Iteratively
            // for (size_t j = 0; j < m.getS(); ++j) {
            //     // g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
            //     predict_belief(j) = g.transpose() * m.getTransitionEndIndex(j) * temp;
            // }
        }

        // t = clock() - t;
        // std::cout << "Time: " << (double) t/CLOCKS_PER_SEC << std::endl;

        // std::cout << cost << std::endl;
        // for (size_t i = 0; i < h*m.getA(); ++i) {
        //  std::cout << gamma[i] << " ";
        // }
        // std::cout << std::endl;

        return cost;
    }

    template<typename M>
    double POMDPSolver::ineq_con(const std::vector<double> &gamma, std::vector<double> &grad, 
        void* cdata) {
        if (!grad.empty()) {
            std::fill(grad.begin(), grad.end(), 0);
            std::vector<double> temp_gamma = gamma;
            std::vector<double> temp;
            for(size_t i = 0; i < gamma.size(); ++i) {
                temp_gamma[i] = std::min(gamma[i] + std::max(gamma[i]*0.05, 0.01), 1.0);
                double adj1 = temp_gamma[i] - gamma[i];
                double temp1 = ineq_con<M>(temp_gamma, temp, cdata);
                temp_gamma[i] = std::max(gamma[i] - std::max(gamma[i]*0.05, 0.01), 0.0);
                double adj2 = temp_gamma[i] - gamma[i];
                double temp2 = ineq_con<M>(temp_gamma, temp, cdata);

                grad[i] = (temp1 - temp2) / (adj1 - adj2);

                temp_gamma[i] = gamma[i];
            }
        }

        OptimizerData<M> *bm = reinterpret_cast<OptimizerData<M>*>(cdata);
        const M & m = bm->model; const Belief & b = bm->belief; double epsilon = bm->epsilon;
        int h = bm->horizon;

        Belief predict_belief = b;
        double vio_rate = 0;
        for (int i = 0; i < h; ++i) {
            // Eigen::Map<Vector> g(gamma.data()+i*h, m.getA());
            Belief temp = predict_belief;
            Vector g(m.getA());
            for (size_t j = 0; j < m.getA(); ++j) 
                g(j) = gamma[i*h+j];

            /*
            // Update belief for each state
            for (size_t j = 0; j < m.getS(); ++j) {
                // trans_end_index_(j): A*S; belief: S*1
                Vector temp_grad = m.getTransitionEndIndex(j) * temp;
                // g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
                predict_belief(j) = g.transpose() * temp_grad;

                if(checkDifferentSmall(predict_belief(j), 0) && m.isViolation(j)) {
                    if (!grad.empty()) {
                        for (size_t k = 0; k < m.getA(); ++k)
                            grad[i*h+k] += temp_grad(k);
                    }
                    vio_rate += predict_belief(j);
                    predict_belief(j) = 0;
                }
            }
            */

            // Update belief 
            //Update together
            predict_belief.setZero();
            for(size_t j = 0; j < m.getA(); ++j) {
                predict_belief.noalias() += Eigen::VectorXd(g(j) * m.getTransitionFunction(j).transpose() * temp);
            }
            // Update iteratively
            // for (size_t j = 0; j < m.getS(); ++j) {
            //     // g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
            //     predict_belief(j) = g.transpose() * m.getTransitionEndIndex(j) * temp;
            // }

            for (size_t j = 0; j < m.getS(); ++j) {
                if(checkDifferentSmall(predict_belief(j), 0) && m.isViolation(j)) {
                    vio_rate += predict_belief(j);
                    predict_belief(j) = 0;
                }
            }
        }

        // return value <= 0 means constraint satisfaction.
        return vio_rate - epsilon;
    }

    template<typename M>
    double POMDPSolver::eq_con(const std::vector<double> &gamma, std::vector<double> &grad, 
        void* cdata) {
        if (!grad.empty()) {
            std::invalid_argument("Optimization solver should be derivative free");
        }

        EqConData<M> *ed = reinterpret_cast<EqConData<M>*>(cdata);
        const M & m = ed->model; int h = ed->horizon; int N = ed->N; 

        if (N >= h) 
            std::invalid_argument("Equality constraints exceed the horizon limit");

        double res = 0;
        for(size_t i = 0; i < m.getA(); ++i) {
            res += gamma[N*m.getA()+i];
        }

        return res-1;

    }

    template<typename M>
    void POMDPSolver::eq_con_vec(unsigned int ms, double* result, unsigned int n, const double* gamma, double* grad, void* cdata) {
        EqConData<M> *ed = reinterpret_cast<EqConData<M>*>(cdata);
        const M & m = ed->model; int h = ed->horizon;

        if (grad) {
            for(size_t i = 0; i < ms; ++i) {
                for(size_t j = 0; j < n; ++j) {
                    grad[i*n+j] = 0;
                    if(j >= i*h && j < (i+1)*h)
                        grad[i*n+j] = 1;
                }
            }
        }

        for(int N = 0; N < h; ++N) {
            double res = 0;
            for(size_t i = 0; i < m.getA(); ++i) {
                res += gamma[N*m.getA()+i];
            }
            result[N] = res - 1.0;
        }

        return;
    }

    template<typename M>
    void POMDPSolver::ineq_con_vec(unsigned int ms, double* result, unsigned int n, const double* gamma, double* grad, void* cdata) {
        EqConData<M> *ed = reinterpret_cast<EqConData<M>*>(cdata);
        const M & m = ed->model; int h = ed->horizon;

        if (grad) {
            for(size_t i = 0; i < ms; ++i) {
                for(size_t j = 0; j < n; ++j) {
                    grad[i*n+j] = 0;
                    if(j >= i*h && j < (i+1)*h)
                        grad[i*n+j] = 1;
                }
            }
        }

        for(int N = 0; N < h; ++N) {
            double res = 0;
            for(size_t i = 0; i < m.getA(); ++i) {
                res += gamma[N*m.getA()+i];
            }
            result[N] = res - 1.0;
        }

        return;
    }

    template<typename M>
    void POMDPSolver::ineq_con_vec_2(unsigned int ms, double* result, unsigned int n, const double* gamma, double* grad, void* cdata) {
        EqConData<M> *ed = reinterpret_cast<EqConData<M>*>(cdata);
        const M & m = ed->model; int h = ed->horizon;

        if (grad) {
            for(size_t i = 0; i < ms; ++i) {
                for(size_t j = 0; j < n; ++j) {
                    grad[i*n+j] = 0;
                    if(j >= i*h && j < (i+1)*h)
                        grad[i*n+j] = -1;
                }
            }
        }

        for(int N = 0; N < h; ++N) {
            double res = 0;
            for(size_t i = 0; i < m.getA(); ++i) {
                res += gamma[N*m.getA()+i];
            }
            result[N] = -res + 1.0;
        }

        return;
    }
}

#endif