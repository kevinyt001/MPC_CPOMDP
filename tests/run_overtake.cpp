
#include <iostream>
#include <fstream>

#include <nlopt.hpp>

#include "Model.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("overtake.POMDP", std::ifstream::in);

	MPC_POMDP::Model overtake = MPC_POMDP::parseCassandra(ifs);

	return 0;
}