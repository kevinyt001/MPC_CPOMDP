#include "Solver.hpp"

namespace MPC_POMDP {
	POMDPSolver::POMDPSolver(const int h, const double e) :
	rand_(Seeder::getSeed())
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
		const Model & m = bm->model; Belief* b = bm->belief; int h = bm->horizon;

		double cost = 0;
		Belief predict_belief = *b;
		for (int i = 0; i < h; ++i) {
			// Eigen::Map<Vector> g(gamma.data()+i*h, m.getA());
			Vector g(m.getA());
			for (size_t j = 0; j < m.getA(); ++j) 
				g(j) = gamma[i*h+j];

			// belief.transpose: 1*S; rewards_: S*A; g(gamma): A*1
			cost += predict_belief.transpose() * m.getRewardFunction() * g;

			Belief temp = predict_belief;

			// Update belief for each state
			for (size_t j = 0; j < m.getS(); ++j) {
				// g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
				predict_belief(j) = g.transpose() * m.getTransitionEndIndex(j) * temp;
			}			
		}

		return cost;
	}

	double POMDPSolver::ineq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, 
		void* cdata) {
		if (!grad.empty()) {
			std::invalid_argument("Optimization solver should be derivative free");
		}

		OptimizerData *bm = reinterpret_cast<OptimizerData*>(cdata);
		const Model & m = bm->model; Belief* b = bm->belief; double epsilon = bm->epsilon;
		int h = bm->horizon;

		Belief predict_belief = *b;
		double vio_rate = 0;
		for (int i = 0; i < h; ++i) {
			// Eigen::Map<Vector> g(gamma.data()+i*h, m.getA());
			Belief temp = predict_belief;
			Vector g(m.getA());
			for (size_t j = 0; j < m.getA(); ++j) 
				g(j) = gamma[i*h+j];

			// Update belief for each state
			for (size_t j = 0; j < m.getS(); ++j) {
				// g.transpose: 1*A; trans_end_index_(j): A*S; belief: S*1
				predict_belief(j) = g.transpose() * m.getTransitionEndIndex(j) * temp;
			}

			for (size_t j = 0; j < m.getS(); ++j) {
				if(checkDifferentSmall(predict_belief(j), 0) && m.isViolation(j)) {
					vio_rate += predict_belief(j);
					predict_belief(j) = 0;
				}
			}
		}

		// return value <= 0 means constraint satisfaction.
		return vio_rate - epsilon;
	}

	double POMDPSolver::eq_constraint(const std::vector<double> &gamma, std::vector<double> &grad, 
		void* cdata) {
		if (!grad.empty()) {
			std::invalid_argument("Optimization solver should be derivative free");
		}

		EqConData *ed = reinterpret_cast<EqConData*>(cdata);
		const Model & m = ed->model; int h = ed->horizon; int N = ed->N; 

		if (N >= h) 
			std::invalid_argument("Equality constraints exceed the horizon limit");

		double res = 0;
		for(size_t i = 0; i < m.getA(); ++i) {
			res += gamma[N*m.getA()+i];
		}

		return res-1;

	}

	void POMDPSolver::operator()(const Model & model, const size_t init_state, Belief& belief) {
		S = model.getS();
		A = model.getA();
		O = model.getO();

		int timestep = 0;

		size_t curr_state = init_state;

		OptimizerData OD = {&belief, model, epsilon_, horizon_};
		
		nlopt::opt opt(nlopt::LN_COBYLA, A*horizon_);
		opt.set_min_objective(cost, &OD);
		opt.add_inequality_constraint(ineq_constraint, &OD, 0.0);

		std::vector<EqConData> eqcon_data(horizon_, {model, horizon_, 0});
		for (int i = 0; i < horizon_; i++) {
			eqcon_data[i].N = i;
			opt.add_equality_constraint(eq_constraint, &eqcon_data[i], equalToleranceGeneral);
		}

		double tot_cost = 0;

		while(!model.isTermination(curr_state)) {
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