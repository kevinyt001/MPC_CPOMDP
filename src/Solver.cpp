#include <Solver.hpp>

namespace MPC_POMDP {
	POMDPSolver::POMDPSolver(const int h, const double e) :
	rand_(getSeed())
	{
		setHorizon(h);
		setEpsilon(e);
	}

	void POMDPSolver::setHorizon(const int h) {
		if ( h < 1 ) throw std::invalid_argument("Horizon must be >= 1");
		horizon_ = h;
	}

	void POMDPSolver::setEpsilon(const double e) {
		if(e > 1 || e < 0) throw std::invalid_argument("Epsilon must be >= 0 and <= 1");
		epsilon_ = e;
	}

	int POMDPSolver::getHorizon() const {
		return horizon_;
	}

	double POMDPSolver::cost(const std::vector<double> &gamma, std::vector<double> &grad, 
		void* fdata) {
		if (!grad.empty()) {
			std::invalid_argument("Solver should be derivative free");
		}

		OptimizerData *bm = reinterpret_cast<OptimizerData*>(fdata);
		Model* m = bm->model; Belief* b = bm->belief;

		double cost = 0;
		Belief predict_belief = *b;
		for (int i = 0; i < horizon_; ++i) {
			Vector g(gamma.data()+i*horizon_, A);

			// belief.transpose: 1*S; rewards_: S*A; g(gamma): A*1
			cost += predict_belief.transpose() * m->getRewardFunction() * g;

			Belief temp = predict_belief;

			// Update belief for each state
			for (size_t j = 0; j < S; ++j) {
				// g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
				predict_belief(j) = g.transpose() * m->getTransitionEndIndex(j) * temp;
			}			
		}

		return cost;
	}

	double ineq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, 
		void* cdata) {
		if (!grad.empty()) {
			std::invalid_argument("Optimization solver should be derivative free");
		}

		OptimizerData *bm = reinterpret_cast<OptimizerData*>(cdata);
		Model* m = bm->model; Belief* b = bm->belief; double epsilon = bm->epsilon;

		double vio_rate = 0;
		for (int i = 0; i < horizon_; ++i) {
			Vector g(gamma.data()+i*horizon_, A);
			Belief temp = predict_belief;

			// Update belief for each state
			for (size_t j = 0; j < S; ++j) {
				// g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
				predict_belief(j) = g.transpose() * m->getTransitionEndIndex(j) * temp;
			}

			for (size_t j = 0; j < S; ++j) {
				if(checkDifferentSmall(predict_belief(j), 0) && m->isViolation(j)) {
					vio_rate += predict_belief(j);
					predict_belief(j) = 0;
				}
			}
		}

		// return value <= 0 means constraint satisfaction.
		return vio_rate - epsilon;
	}

	double eq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, 
		void* cdata) {
		if (!grad.empty()) {
			std::invalid_argument("Optimization solver should be derivative free");
		}

		int *N = reinterpret_cast<int*>(cdata);

		if (*N >= horizon_) 
			std::invalid_argument("Equality constraints exceed the horizon limit");

		double res = 0;
		for(size_t i = 0; i < A; ++i) {
			res += gamma[(*N)*A+i];
		}

		return res-1;

	}

	void POMDPSolver::operator()(const Model & model, const size_t init_state, Belief& belief) {
		S = model.getS();
		A = model.getA();
		O = model.getO();

		int timestep = 0;

		size_t curr_state = init_state;

		OptimizerData OD = {&belief, &model, &epsilon_};
		
		nlopt::opt opt(nlopt::LN_COBYLA, A*horizon_);
		opt.set_min_objective(cost, &OD);
		opt.add_inequality_constraint(ineq_constraint, &OD, 0.0);

		std::vector<int> eqcon_data(horizon_, 0);
		for (int i = 0; i < horizon_; i++) {
			eqcon_data[i] = i;
			opt.add_equality_constraint(eq_constraint, &eqcon_data[i], equalToleranceGeneral);
		}

		double tot_cost = 0;

		while(!model.isTerminal(curr_state)) {
			std::vector<double> gamma(A*horizon_, 1.0/(double)A);
			double cost_temp = 0;

			try{
				nlopt::result result = opt.optimize(gamma, cost_temp);
			}
			catch(std::exception &e) {
    			std::cout << "nlopt failed: " << e.what() << std::endl;
			}

			tot_cost += cost_temp;

			size_t action = sampleProbability(A, gamma, rand_);

			std::tuple<size_t, size_t, double> SOR = model.propagateSOR(curr_state, action);

			Belief belief_temp(S);
			updateBelief(model, belief, action, std::get<1>(SOR), &belief_temp);
			belief = belief_temp;

			curr_state = std::get<0>(SOR);

		}
	}
}