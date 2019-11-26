
// #define EIGEN_USE_MKL_ALL

#include <iostream>
#include <fstream>

#include <nlopt.hpp>

// #include "Model.hpp"
#include "models/SparseModel.hpp"
#include "SolverNLOPT.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/CDC19_intersection.POMDP", std::ifstream::in);

	MPC_POMDP::SparseModel overtake = MPC_POMDP::parseCassandraSparse(ifs);
	// MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	int horizon = 3;
	double epsilon = 0.01;
	MPC_POMDP::POMDPSolver_NLOPT solver(horizon, epsilon);

	size_t init_state = 3969;
	size_t init_state_2 = 0;
	// MPC_POMDP::SparseBelief belief(overtake.getS());
	MPC_POMDP::Belief belief(overtake.getS());
	belief.setZero();

	belief(init_state) = 0.5;
	belief(init_state_2) = 0.5;

	solver(overtake, init_state, belief);

	return 0;
}