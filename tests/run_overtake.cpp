
// #define EIGEN_USE_MKL_ALL

#include <iostream>
#include <fstream>

#include <nlopt.hpp>

#include "Model.hpp"
#include "SparseModel.hpp"
#include "Solver.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/overtake.POMDP", std::ifstream::in);

	MPC_POMDP::SparseModel overtake = MPC_POMDP::parseCassandraSparse(ifs);
	// MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	int horizon = 5;
	double epsilon = 0.05;
	MPC_POMDP::POMDPSolver solver(horizon, epsilon);

	size_t init_state = 328;
	MPC_POMDP::SparseBelief belief(overtake.getS());
	// MPC_POMDP::Belief belief(overtake.getS());
	belief.setZero();

	belief.insert(init_state) = 1.0;

	solver(overtake, init_state, belief);

	return 0;
}