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

}