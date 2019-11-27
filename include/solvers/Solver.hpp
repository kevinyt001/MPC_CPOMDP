#ifndef MPC_POMDP_SOLVER_HEADER_FILE
#define MPC_POMDP_SOLVER_HEADER_FILE

#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <nlopt.hpp>

#include "models/DenseModel.hpp"
#include "models/SparseModel.hpp"
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

            virtual ~POMDPSolver() {};

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

        protected:
        	size_t S, A, O;
        	int horizon_;
        	double epsilon_;

        	mutable RandomEngine rand_;
	};
}

#endif