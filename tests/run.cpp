
#include "utilities/CassandraParser.hpp"
#include "utilities/Core.hpp"
#include "IO.hpp"
#include "Model.hpp"
#include "SparseModel.hpp"
#include "SolverIPOPT.hpp"
#include "SolverNLOPT.hpp"

#include <iostream>
#include <fstream>

int main(int ac, char* av[]) {
	
	read_parameters(ac,av);

	if (params::model_type == "dense" && params::model_scale == "large") {
		throw std::invalid_argument("Dense model cannot be large scale due to memory limitations.");
	}

}
