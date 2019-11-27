
#include "utilities/CassandraParser.hpp"
#include "utilities/Core.hpp"
#include "utilities/parameter_reader.hpp"
#include "IO.hpp"
#include "models/DenseModel.hpp"
#include "models/SparseModel.hpp"
#include "solvers/SolverIPOPT.hpp"
#include "solvers/SolverNLOPT.hpp"

#include <iostream>
#include <fstream>

int main(int ac, char* av[]) {
	
	read_parameters(ac,av);

	if(params::model_type == "dense" && params::model_scale == "large") {
		throw std::invalid_argument("Dense model cannot be large scale due to memory limitations.");
	}

	MPC_POMDP::DenseModel* dense_model;
	MPC_POMDP::SparseModel* sparse_model;

	std::ifstream ifs;
	ifs.open(params::file_name, std::ifstream::in);
	if(params::model_type == "dense") {
		if(params::model_scale == "small") {
			dense_model = MPC_POMDP::parseCassandra(ifs);
			sparse_model = NULL;
		}
	}
	else if(params::model_type == "sparse") {
		if(params::model_scale == "small") {
			sparse_model = MPC_POMDP::parseCassandraSparse(ifs);
			dense_model = NULL;
		}
		else if(params::model_scale == "large") {
			sparse_model = MPC_POMDP::parseCassandraLarge(ifs);
			dense_model = NULL;
		}
	}
	ifs.close();

	MPC_POMDP::POMDPSolver_IPOPT* solver_ipopt;
	MPC_POMDP::POMDPSolver_NLOPT* solver_nlopt;
	if(params::solver_type == "nlopt") {
		solver_nlopt = new MPC_POMDP::POMDPSolver_NLOPT(params::horizon, params::epsilon);
		solver_ipopt = NULL;
	}
	else if(params::solver_type == "ipopt") {
		solver_ipopt = new MPC_POMDP::POMDPSolver_IPOPT(params::horizon, params::epsilon);
		solver_nlopt = NULL;
	}

	size_t S = 0;
	if(dense_model) S = dense_model->getS();
	if(sparse_model) S = sparse_model->getS();

	MPC_POMDP::Belief belief(S);
	belief.setZero();
	for(size_t i = 0; i < params::start_belief.size(); ++i) {
		belief(params::start_belief[i]) = 1.0/(double) params::start_belief.size();
	}

	if(solver_nlopt && dense_model) {
		solver_nlopt->operator()(*dense_model, params::start_state, belief);
	}
	if(solver_ipopt && dense_model) solver_ipopt->operator()(*dense_model, params::start_state, belief);
	if(solver_nlopt && sparse_model) solver_nlopt->operator()(*sparse_model, params::start_state, belief);
	if(solver_ipopt && sparse_model) solver_ipopt->operator()(*sparse_model, params::start_state, belief);
	
	if(dense_model) delete dense_model;
	if(sparse_model) delete sparse_model;
	if(solver_nlopt) delete solver_nlopt;
	if(solver_ipopt) delete solver_ipopt;

}
