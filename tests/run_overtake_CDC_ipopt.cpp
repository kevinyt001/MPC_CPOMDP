
// #define EIGEN_USE_MKL_ALL

#include <iostream>
#include <fstream>

#include <nlopt.hpp>

#include "Model.hpp"
#include "SparseModel.hpp"
#include "Solver.hpp"
#include "SolverIPOPT.hpp"
#include "IpIpoptApplication.hpp"
#include "POMDP_NLP.hpp"
#include "IpTNLP.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/CDC19_overtake.POMDP", std::ifstream::in);

	MPC_POMDP::SparseModel overtake = MPC_POMDP::parseCassandraSparse(ifs);
	// MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	int horizon = 3;
	double epsilon = 0.01;
	MPC_POMDP::POMDPSolver_IPOPT solver(horizon, epsilon);

	size_t init_state = 1360;
	size_t init_state_2 = 136;
	// MPC_POMDP::SparseBelief belief(overtake.getS());
	MPC_POMDP::Belief belief(overtake.getS());
	belief.setZero();

	belief(init_state) = 0.5;
	belief(init_state_2) = 0.5;

	solver(overtake, init_state, belief);

	return 0;
}