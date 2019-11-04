
#include <iostream>
#include <fstream>

#include <nlopt.hpp>

#include "Model.hpp"
#include "Solver.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/overtake.POMDP", std::ifstream::in);

	MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	int horizon = 5;
	double epsilon = 0.05;
	MPC_POMDP::POMDPSolver solver(horizon, epsilon);

	MPC_POMDP::Belief belief(overtake.getS());
	belief.setZero();

	size_t init_state = 328;

	belief(init_state) = 1.0;

	solver(overtake, init_state, belief);

	return 0;
}