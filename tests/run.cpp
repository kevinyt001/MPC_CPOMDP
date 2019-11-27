
#include "utilities/CassandraParser.hpp"
#include "utilities/Core.hpp"
#include "utilities/parameter_reader.hpp"
#include "IO.hpp"
#include "models/DenseModel.hpp"
#include "models/SparseModel.hpp"
#include "SolverIPOPT.hpp"
#include "SolverNLOPT.hpp"

#include <iostream>
#include <fstream>

int main(int ac, char* av[]) {
	
	read_parameters(ac,av);

	if(params::model_type == "dense" && params::model_scale == "large") {
		throw std::invalid_argument("Dense model cannot be large scale due to memory limitations.");
	}

	MPC_POMDP::Model* model;
	if(params::model_type == "dense") {
		if(params::model_scale == "small") {
			std::ifstream ifs;
			ifs.open(params::file_name, std::ifstream::in);
			*model = MPC_POMDP::parseCassandra(ifs);
		}
	}

	std::cout << model->getS() << std::endl;

}
