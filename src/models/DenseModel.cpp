#include "Model.hpp"

namespace MPC_POMDP{

	DenseModel::DenseModel(const size_t s, const size_t a, const size_t o, const double discount):
            Model(s, a, o, d),
            transitions_(A, Matrix2D(S, S)), trans_end_index_(S, Matrix2D(A, S)), 
            rewards_(S, A), observations_(A, Matrix2D(S, O))
	{
		for (size_t a = 0; a < A; a++)
			transitions_[a].setIdentity();

		rewards_.setZero();

		for (size_t a = 0; a < A; a++) {
			observations_[a].rightCols(O-1).setZero();
			observations_[a].col(0).fill(1.0);
		}

        updateTransEndIndex();

	}

	DenseModel::DenseModel(const DenseModel& model): 
            Model(model.getS(), model.getA(), model.getO()),
            transitions_(A, Matrix2D(S, S)), trans_end_index_(S, Matrix2D(A, S)), 
            rewards_(S, A), observations_(A, Matrix2D(S, O))
	{
		setDiscount(model.getDiscount());
		rewards_.setZero();
		for (size_t a = 0; a < A; ++a) {
			for (size_t s = 0; s < S; ++s) {
				//Copy Transition and Reward Matrices
				for (size_t s1 = 0; s1 < S; ++s1) {
					transitions_[a](s, s1) = model.getTransitionProbability(s, a, s1);
					rewards_(s, a) += model.getExpectedReward(s, a, s1) * transitions_[a](s, s1); //This line may not be correct
				}
                if ( !isProbability(S, transitions_[a].row(s)) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
                
                //Copy Observation Matrix
                for (size_t o = 0; o < O; ++o)
                {
 					observations_[a](s, o) = model.getObservationProbability(s, a, o);
                }
				if ( !isProbability(O, observations_[a].row(s)) )
                    throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");
			}
		}
        
        setTerminationFunction(model.getTerminationFunction());
        setViolationFunction(model.getViolationFunction());

        updateTransEndIndex();

	}

    DenseModel::DenseModel(NoCheck, const size_t s, const size_t a, const size_t o, TransitionMatrix && t, 
    	RewardMatrix && r, ObservationMatrix && om, std::vector<bool>& ter, 
        std::vector<bool>& vio, const double d):
            Model(s, a, o, d),
    		transitions_(std::move(t)), trans_end_index_(S, Matrix2D(A, S)), 
            rewards_(std::move(r)), observations_(om)
    {
        setTerminationFunction(ter);
        setViolationFunction(vio);

        updateTransEndIndex();
    }

    void DenseModel::setTransitionFunction(const TransitionMatrix & t) {
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s = 0; s < S; ++s ) {
                if ( t[a].row(s).minCoeff() < 0.0 ||
                     !checkEqualSmall(1.0, t[a].row(s).sum()) )
                {
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
                }
            }
        }
        
        transitions_ = t;
        updateTransEndIndex();
    }

    void DenseModel::setRewardFunction(const RewardMatrix & r) {
        rewards_ = r;
    }

	void DenseModel::setObservationFunction(const ObservationMatrix & om) {
		for (size_t a = 0; a < A; ++a) {
			for (size_t s = 0; s < S; ++s) {
				if ( !isProbability(O, om[a].row(s)) )
                    throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");
			}
		}

		observations_ = om;
	}

    const DenseModel::TransitionMatrix & DenseModel::getTransitionFunction() const { return transitions_; }
    const DenseModel::TransitionMatrix & DenseModel::getTransitionEndIndex() const { return trans_end_index_; }
    const DenseModel::RewardMatrix &     DenseModel::getRewardFunction()     const { return rewards_; }
    const DenseModel::ObservationMatrix & DenseModel::getObservationFunction() const { return observations_; }

    const Matrix2D & DenseModel::getTransitionFunction(const size_t a) const { return transitions_[a]; }
    const Matrix2D & DenseModel::getTransitionEndIndex(const size_t s1) const { return trans_end_index_[s1]; }
    const Matrix2D & DenseModel::getObservationFunction(const size_t a) const { return observations_[a]; }

    std::tuple<size_t,size_t, double> DenseModel::propagateSOR(const size_t s, const size_t a) const {
        const size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);
        const double r = rewards_(s, a);
        const auto o = sampleProbability(O, observations_[a].row(s1), rand_);
        return std::make_tuple(s1, o, r);
    }

    void DenseModel::updateTransEndIndex() {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    trans_end_index_[s1](a, s) = transitions_[a](s, s1);
    }
}