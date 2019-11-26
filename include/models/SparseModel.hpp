#ifndef MPC_POMDP_SPARSE_MODEL_HEADER_FILE
#define MPC_POMDP_SPARSE_MODEL_HEADER_FILE

#include <utility>
#include <random>
#include <vector>

#include "DefTypes.hpp"
#include "models/Model.hpp"
#include "utilities/Seeder.hpp"
#include "utilities/Probability.hpp"

namespace MPC_POMDP {
	class SparseModel: public Model {
        public:
    		using TransitionMatrix = SparseMatrix3D;
    		using RewardMatrix = SparseMatrix2D;
    		using ObservationMatrix = SparseMatrix3D;

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the SparseModel so that all
             * transitions happen with probability 0 but for transitions
             * that bring back to the same state, no matter the action.
             *
             * All rewards are set to 0. Initialize obervations that all
             * actions will return observation 0.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param o The number of observations the agent could make.
             * @param discount The discount factor for the POMDP.
             */
    		SparseModel(size_t s, size_t a, size_t o, double discount = 1.0);

            /**
             * @brief Copy constructor from any valid POMDP model.
             *
             * This allows to copy from any other model. A nice use for this is to
             * convert any model which computes probabilities on the fly into a
             * Model where probabilities are all stored for fast access. Of
             * course such a solution can be done only when the number of states
             * and actions is not too big.
             *
             * @param model The model that needs to be copied.
             */
            SparseModel(const SparseModel& model);

            /**
             * @brief Basic constructor.
             *
             * This constructor takes three arbitrary three dimensional
             * containers and tries to copy their contents into the
             * transitions, rewards and observations matrices respectively.
             *
             * The containers need to support data access through
             * operator[]. In addition, the dimensions of the containers
             * must match the ones provided as arguments (for three
             * dimensions: s,a,s).
             *
             * This is important, as this constructor DOES NOT perform any
             * size checks on the external containers.
             *
             * Internal values of the containers will be converted to
             * double, so these conversions must be possible.
             *
             * In addition, the transition container must contain a valid
             * transition function.
             *
             * The discount parameter must be between 0 and 1 included,
             * otherwise the constructor will throw an std::invalid_argument.
             *
             * @tparam T  The external transition container type.
             * @tparam R  The external rewards container type.
             * @tparam OM The external observation container type.
             * @tparam TER The external termination container type.
             * @tparam VIO The external violation container type.
             * @param s  The number of states of the world.
             * @param a  The number of actions available to the agent.
             * @param o  The number of observations available to the agent.
             * @param t  The external transitions container.
             * @param r  The external rewards container.
             * @param om The external observations container.
             * @param ter The external terminations container.
             * @param vio The external violations container.
             * @param d  The discount factor for the POMDP.
             */
            template <typename T, typename R, typename OM, typename TER, typename VIO>
            SparseModel(size_t s, size_t a, size_t o, const T & t, const R & r, 
            	const OM & om, const TER & ter, const VIO & vio, double d = 1.0);

            /**
             * @brief Unchecked constructor.
             *
             * This constructor takes ownership of the data that it is passed
             * to it to avoid any sorts of copies and additional work (sanity
             * checks), in order to speed up as much as possible the process of
             * building a new Model.
             *
             * Note that to use it you have to explicitly use the NO_CHECK tag
             * parameter first.
             *
             * @param s  The number of states of the world.
             * @param a  The number of actions available to the agent.
             * @param o  The number of observations available to the agent.
             * @param t  The transition function to be used in the Model.
             * @param r  The reward function to be used in the Model.
             * @param om The observation function to be used in the Model.
             * @param d  The discount factor for the Model.
             */
            SparseModel(NoCheck, size_t s, size_t a, size_t o, TransitionMatrix && t, 
            	RewardMatrix && r, ObservationMatrix && om, std::vector<bool> & ter,
                  std::vector<bool> & vio, double d);
            
            SparseModel(NoCheck, size_t s, size_t a, size_t o, const TransitionMatrix & t, 
                  const RewardMatrix & r, const ObservationMatrix & om, const std::vector<bool> & ter,
                  const std::vector<bool> & vio, double d);

            /**
             * @brief This function replaces the SparseModel transition function with the one provided.
             *
             * This function will throw a std::invalid_argument if the
             * matrix provided does not contain valid probabilities.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the container
             * must match the ones provided as arguments (for three
             * dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform any
             * size checks on the external container.
             *
             * Internal values of the container will be converted to
             * double, so these conversions must be possible.
             *
             * @tparam T The external transition container type.
             * @param t The external transitions container.
             */
            template <typename T>
            void setTransitionFunction(const T & t);

            /**
             * @brief This function sets the transition function using a Eigen sparse matrices.
             *
             * This function will throw a std::invalid_argument if the
             * matrix provided does not contain valid probabilities.
             *
             * The dimensions of the container must match the ones provided
             * as arguments (for three dimensions: A, S, S). BE CAREFUL.
             * The sparse matrices MUST be SxS, while the std::vector
             * containing them MUST represent A.
             *
             * This function does DOES NOT perform any size checks on the
             * input.
             *
             * @param t The external transitions container.
             */
            void setTransitionFunction(const TransitionMatrix & t);

            /**
             * @brief This function replaces the Model reward function with the one provided.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the containers
             * must match the ones provided as arguments (for three
             * dimensions: S,A,S).
             *
             * This is important, as this function DOES NOT perform any
             * size checks on the external containers.
             *
             * Internal values of the container will be converted to
             * double, so these conversions must be possible.
             *
             * @tparam R The external rewards container type.
             * @param r The external rewards container.
             */
            template <typename R>
            void setRewardFunction(const R & r);

            /**
             * @brief This function replaces the reward function with the one provided.
             *
             * The dimensions of the container must match the ones provided
             * as arguments (for two dimensions: S, A). BE CAREFUL.
             *
             * This function does DOES NOT perform any size checks on the
             * input.
             *
             * @param r The external rewards container.
             */
            void setRewardFunction(const RewardMatrix & r);

            /**
             * @brief This function replaces the Model observation function with the one provided.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the
             * containers must match the ones provided as arguments
             * (for three dimensions: s,a,o, in this order).
             *
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * Internal values of the container will be converted to double,
             * so these conversions must be possible.
             *
             * @tparam OM The external observations container type.
             * @param om The external observations container.
             */
            template <typename OM>
            void setObservationFunction(const OM & om);

            /**
             * @brief This function replaces the Model observation function with the one provided.
             *
             * The dimensions of the container must match the ones provided
             * as arguments (for three dimensions: A, S, O). BE CAREFUL.
             *
             * This function does DOES NOT perform any size checks on the
             * input.
             *
             * @param om The external observation container.
             */
            void setObservationFunction(const ObservationMatrix & om);

            /**
             * @brief This function returns the stored transition probability for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The probability of the specified transition.
             */
            double getTransitionProbability(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the stored expected reward for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The expected reward of the specified transition.
             */
            double getExpectedReward(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the stored observation probability for the specified state-action pair.
             *
             * @param s1 The final state of the transition.
             * @param a The action performed in the transition.
             * @param o The recorded observation for the transition.
             *
             * @return The probability of the specified observation.
             */
            double getObservationProbability(size_t s1, size_t a, size_t o) const;
            
            /**
             * @brief This function returns the transition matrix for inspection.
             *
             * @return The transition matrix.
             */
            const TransitionMatrix & getTransitionFunction() const;

            /**
             * @brief This function returns the transition function for a given action.
             *
             * @param a The action requested.
             *
             * @return The transition function for the input action.
             */
            const SparseMatrix2D & getTransitionFunction(size_t a) const;

            /**
             * @brief This function returns the trans_end_index matrix for inspection.
             *
             * @return The trans_end_index matrix.
             */
            const TransitionMatrix & getTransitionEndIndex() const;

            /**
             * @brief This function returns the trans_end_index for a given end state.
             *
             * @param s1 The end state requested.
             *
             * @return The transition function for the input action.
             */
            const SparseMatrix2D & getTransitionEndIndex(size_t s1) const;

            /**
             * @brief This function returns the rewards matrix for inspection.
             *
             * @return The rewards matrix.
             */
            const RewardMatrix & getRewardFunction() const;

            /**
             * @brief This function returns the observation matrix for inspection.
             *
             * @return The observation matrix.
             */
            const ObservationMatrix & getObservationFunction() const;

            /**
             * @brief This function returns the observation function for a given action.
             *
             * @param a The action requested.
             *
             * @return The observation function for the input action.
             */
            const SparseMatrix2D & getObservationFunction(size_t a) const;

            /**
            * @brief This function propogates the POMDP for the specified state action pair.
            *
            * This function propagates the model for simulated experience. The
            * transition, observation and reward functions are used to
            * produce, from the state action pair inserted as arguments, a
            * possible new state with respective observation and reward.
            * The new state is picked from all possible states that the
            * POMDP allows transitioning to, each with probability equal to
            * the same probability of the transition in the model. After a
            * new state is picked, an observation is sampled from the
            * observation function distribution, and finally the reward is
            * the corresponding reward contained in the reward function.
            *
            * The rewards returned is the expected reward from state s and take action a.
            *
            * @param s The state that needs to be sampled.
            * @param a The action that needs to be sampled.
            *
            * @return A tuple containing a new state, observation and reward.
            */
            std::tuple<size_t, size_t, double> propagateSOR(size_t s,size_t a) const;

        private:      
            // Contain the transition probability from s0 to s1 by action a
            // with transitions_[a](s0, s1)
            TransitionMatrix transitions_;
            // Restructure the transition matrix so that it is indexed by end state
            // i.e. trans_end_index[s1](A, s0) with size S*A*S
            TransitionMatrix trans_end_index_; 
            // Contain the expected reward at state s with action a
            RewardMatrix rewards_; 
            // Observation Matrix for each action is a probability distribution
            ObservationMatrix observations_; 

            /**
            * @brief This function updates trans_end_index according to transitions_
            */
            void updateTransEndIndex();
	};

    template <typename T, typename R, typename OM, typename TER, typename VIO>
    SparseModel::SparseModel(const size_t s, const size_t a, const size_t o, const T & t, const R & r, 
        const OM & om, const TER & ter, const VIO & vio, const double d):
            Model(s, a, o, d),
            transitions_(A, SparseMatrix2D(S, S)), trans_end_index_(S, SparseMatrix2D(A, S)),
            rewards_(S, A), observations_(A, SparseMatrix2D(S, O))
    {
        setTransitionFunction(t);
        setRewardFunction(r);
        setObservationFunction(om);
        setTerminationFunction(ter);
        setViolationFunction(vio);
    }

    template <typename T>
    void SparseModel::setTransitionFunction(const T & t) {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                if ( !isProbability(S, t[s][a]) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");

        for ( size_t a = 0; a < A; ++a ) {
            transitions_[a].setZero();

            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    const double p = t[s][a][s1];
                    if ( checkDifferentSmall(0.0, p) ) transitions_[a].insert(s, s1) = p;
            }
            transitions_[a].makeCompressed();
        }

        updateTransEndIndex();
    }

    template <typename R>
    void SparseModel::setRewardFunction(const R & r) {
        rewards_.setZero();
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 ) {
                    const double w = r[s][a][s1];
                    const double p = transitions_[a].coeff(s, s1);
                    if ( checkDifferentSmall(0.0, w) && checkDifferentSmall(0.0, p) ) rewards_.coeffRef(s, a) += w * p;
            }
        }
        rewards_.makeCompressed();
    }

    template <typename OM>
    void SparseModel::setObservationFunction(const OM & om) {
        for ( size_t s1 = 0; s1 < S; ++s1 )
            for ( size_t a = 0; a < A; ++a )
                if ( !isProbability(O, om[s1][a]) )
                    throw std::invalid_argument("Input observation matrix does not contain valid probabilities.");

        for ( size_t a = 0; a < A; ++a ) {
            observations_[a].setZero();

            for ( size_t s1 = 0; s1 < S; ++s1 )
                for ( size_t o = 0; o < O; ++o ) {
                    const double p = om[s1][a][o];
                    if ( checkDifferentSmall(0.0, p) ) observations_[a].insert(s1, o) = p;
            }
            observations_[a].makeCompressed();
        }
    }
}


#endif