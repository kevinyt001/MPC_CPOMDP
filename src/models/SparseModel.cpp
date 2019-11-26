#include "SparseModel.hpp"

#include <fstream>
#include <iostream>

namespace MPC_POMDP{

	SparseModel::SparseModel(const size_t s, const size_t a, const size_t o, const double discount):
            Model(s, a, o, d),
            transitions_(A, SparseMatrix2D(S, S)), trans_end_index_(S, SparseMatrix2D(A, S)), 
            rewards_(S, A), observations_(A, SparseMatrix2D(S, O))
	{
		for (size_t a = 0; a < A; a++)
			transitions_[a].setIdentity();

		rewards_.setZero();

        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s1 = 0; s1 < S; ++s1 )
                observations_[a].insert(s1, 0) = 1.0;
            observations_[a].makeCompressed();
        }

        updateTransEndIndex();

	}

	SparseModel::SparseModel(const SparseModel& model):
            Model(model.getS(), model.getA(), model.getO()),
            transitions_(A, SparseMatrix2D(S, S)), trans_end_index_(S, SparseMatrix2D(A, S)), 
            rewards_(S, A), observations_(A, SparseMatrix2D(S, O))
	{
		setDiscount(model.getDiscount());
		rewards_.setZero();
		for (size_t a = 0; a < A; ++a) {
			for (size_t s = 0; s < S; ++s) {
				//Copy Transition and Reward Matrices
				for (size_t s1 = 0; s1 < S; ++s1) {
                    const double p = model.getTransitionProbability(s, a, s1);
                    if ( p < 0.0 || p > 1.0 )
                        throw std::invalid_argument("Input transition matrix contains an invalid value.");

                    if ( checkDifferentSmall(0.0, p) ) transitions_[a].insert(s, s1) = p;
                    const double r = model.getExpectedReward(s, a, s1);
                    if ( checkDifferentSmall(0.0, r) ) rewards_.coeffRef(s, a) += r * p;
				}
                if ( checkDifferentSmall(1.0, transitions_[a].row(s).sum()) )
                    throw std::invalid_argument("Input transition matrix contains an invalid row.");
                
                //Copy Observation Matrix
                for (size_t o = 0; o < O; ++o) {
                    const double p = model.getObservationProbability(s, a, o);
                    if ( p < 0.0 || p > 1.0 )
                        throw std::invalid_argument("Input observation matrix contains an invalid value.");

                    if ( checkDifferentSmall(0.0, p) ) observations_[a].insert(s, o) = p;
                }
                if ( checkDifferentSmall(1.0, observations_[a].row(s).sum()) )
                    throw std::invalid_argument("Input observation matrix contains an invalid row.");
			}
		}

        for ( size_t a = 0; a < A; ++a ) {
            transitions_[a].makeCompressed();
            observations_[a].makeCompressed();
        }
        rewards_.makeCompressed();
        
        setTerminationFunction(model.getTerminationFunction());
        setViolationFunction(model.getViolationFunction());

        updateTransEndIndex();

	}

    SparseModel::SparseModel(NoCheck, const size_t s, const size_t a, const size_t o, TransitionMatrix && t, 
    	RewardMatrix && r, ObservationMatrix && om, std::vector<bool>& ter, 
        std::vector<bool>& vio, const double d):
            Model(s, a, o, d),
    		transitions_(std::move(t)), trans_end_index_(S, SparseMatrix2D(A, S)),
    		rewards_(std::move(r)), observations_(om)
    {
        setTerminationFunction(ter);
        setViolationFunction(vio);

        updateTransEndIndex();
    }

    SparseModel::SparseModel(NoCheck, const size_t s, const size_t a, const size_t o, const TransitionMatrix & t, 
        const RewardMatrix & r, const ObservationMatrix & om, const std::vector<bool>& ter, 
        const std::vector<bool>& vio, const double d):
            Model(s, a, o, d),
            transitions_(t), trans_end_index_(S, SparseMatrix2D(A, S)),
            rewards_(r), observations_(om)
    {
        // std::cout << "Start Constructing Model" << std::endl;
        if(!rewards_.isCompressed()) rewards_.makeCompressed();
        // std::cout << "Compressed Rewards" << std::endl;
        for(size_t i = 0; i < A; ++i) {
            // std::cout << "i: " << i << std::endl;
            if(!transitions_[i].isCompressed()) transitions_[i].makeCompressed();
            if(!observations_[i].isCompressed()) observations_[i].makeCompressed();
        }
        // std::cout << "Compressed transitions and observations"<< std::endl;
        setTerminationFunction(ter);
        setViolationFunction(vio);
        
        updateTransEndIndex();
        // std::cout << "Finish updating trans_end_index_" << std::endl;
    }

    void SparseModel::setTransitionFunction(const TransitionMatrix & t) {
        // Eigen sparse does not implement minCoeff so we can't check for negatives.
        // So we force the matrix to its abs, and if then the sum goes haywire then
        // we found an error.
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s = 0; s < S; ++s ) {
                if ( !checkEqualSmall(1.0, t[a].row(s).sum()) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
                if ( !checkEqualSmall(1.0, t[a].row(s).cwiseAbs().sum()) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
            }
        }
        
        transitions_ = t;
        updateTransEndIndex();
    }

    void SparseModel::setRewardFunction(const RewardMatrix & r) {
        rewards_ = r;
    }

	void SparseModel::setObservationFunction(const ObservationMatrix & om) {
		for (size_t a = 0; a < A; ++a) {
			for (size_t s = 0; s < S; ++s) {
                if ( !checkEqualSmall(1.0, om[a].row(s).sum()) )
                    throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");
                if ( !checkEqualSmall(1.0, om[a].row(s).cwiseAbs().sum()) )
                    throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");
			}
		}

		observations_ = om;
	}


    double SparseModel::getTransitionProbability(const size_t s, const size_t a, const size_t s1) const {
        return transitions_[a].coeff(s, s1);
    }

    double SparseModel::getExpectedReward(const size_t s, const size_t a, const size_t) const {
        return rewards_.coeff(s, a);
    }

    double SparseModel::getObservationProbability(const size_t s1, const size_t a, const size_t o) const {
        return observations_[a].coeff(s1, o);
    }

    const SparseModel::TransitionMatrix & SparseModel::getTransitionFunction() const { return transitions_; }
    const SparseModel::TransitionMatrix & SparseModel::getTransitionEndIndex() const { return trans_end_index_; }
    const SparseModel::RewardMatrix &     SparseModel::getRewardFunction()     const { return rewards_; }
    const SparseModel::ObservationMatrix & SparseModel::getObservationFunction() const { return observations_; }

    const SparseMatrix2D & SparseModel::getTransitionFunction(const size_t a) const { return transitions_[a]; }
    const SparseMatrix2D & SparseModel::getTransitionEndIndex(const size_t s1) const { return trans_end_index_[s1]; }
    const SparseMatrix2D & SparseModel::getObservationFunction(const size_t a) const { return observations_[a]; }

    std::tuple<size_t,size_t, double> SparseModel::propagateSOR(const size_t s, const size_t a) const {
        const size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);
        const double r = rewards_.coeff(s, a);
        const size_t o = sampleProbability(O, observations_[a].row(s1), rand_);
        return std::make_tuple(s1, o, r);
    }

    void SparseModel::updateTransEndIndex() {
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            trans_end_index_[s1].setZero();
            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t s = 0; s < S; ++s ) {
                    const double p = transitions_[a].coeff(s, s1);
                    if ( checkDifferentSmall(0.0, p) ) trans_end_index_[s1].insert(a, s) = p;
                }
            }
        }
    }
}