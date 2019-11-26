#include "Model.hpp"

namespace MPC_POMDP{

	Model::Model(const size_t s, const size_t a, const size_t o, const double discount):
            S(s), A(a), O(o), discount_(discount), 
            terminations_(S, true), violations_(S, false), rand_(Seeder::getSeed())
	{}

    void Model::setTerminationFunction(const std::vector<bool> & ter) {
        if (ter.size() != S) throw std::invalid_argument("Input termination function does not have the correct size");
        terminations_ = ter;
    }

    void Model::setViolationFunction(const std::vector<bool> & vio) {
        if (vio.size() != S) throw std::invalid_argument("Input violation function does not have the correct size");
        violations_ = vio;
    }

    void Model::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

	size_t Model::getS() const { return S; }
	size_t Model::getA() const { return A; }
	size_t Model::getO() const { return O; }
	double Model::getDiscount() const { return discount_; }

    const std::vector<bool> & Model::getTerminationFunction() const { return terminations_; }
    const std::vector<bool> & Model::getViolationFunction() const { return violations_; }

    bool Model::isTermination(const size_t s) const { return terminations_[s]; }
    bool Model::isViolation(const size_t s) const { return violations_[s]; }
}