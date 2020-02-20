
// #define EIGEN_USE_MKL_ALL

#include <iostream>
#include <fstream>

#include <nlopt.hpp>

#include "Model.hpp"
#include "SparseModel.hpp"
#include "SolverNLOPT.hpp"
#include "SolverIPOPT.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/CDC19_merge.POMDP", std::ifstream::in);

	MPC_POMDP::SparseModel overtake = MPC_POMDP::parseCassandraLarge(ifs);
	// MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	std::cout << "Finish constructing the model" << std::endl;

	int horizon = 3;
	double epsilon = 0.01;
	MPC_POMDP::POMDPSolver_IPOPT solver(horizon, epsilon);

	size_t init_state = 2688;
	size_t init_state_2 = 10626;

	for (int i = 0; i < 30; i++) {
		// MPC_POMDP::SparseBelief belief(overtake.getS());
		MPC_POMDP::Belief belief(overtake.getS());
		belief.setZero();

		belief(init_state) = 0.5;
		belief(init_state_2) = 0.5;

		// std::cout << "Start solving ..." << std::endl;

		// std::cout << "Model S:" << overtake.getS() << std::endl;
		// std::cout << "Model A:" << overtake.getA() << std::endl;
		// std::cout << "Model O:" << overtake.getO() << std::endl;

		// std::cout << "Model transition size:" << overtake.getTransitionFunction().size()
		// << " x " << overtake.getTransitionFunction()[0].rows() << " x " <<
		// overtake.getTransitionFunction()[0].cols() << std::endl;

		// std::cout << "Model rewards size:" << overtake.getRewardFunction().rows() << " x " <<
		// overtake.getRewardFunction().cols() << std::endl;

		// std::cout << "Model observation size:" << overtake.getObservationFunction().size()
		// << " x " << overtake.getObservationFunction()[0].rows() << " x " <<
		// overtake.getObservationFunction()[0].cols() << std::endl;

		solver(overtake, init_state, belief);
	}

	return 0;
}