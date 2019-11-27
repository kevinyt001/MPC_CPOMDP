#ifndef MPC_POMDP_MODEL_HEADER_FILE
#define MPC_POMDP_MODEL_HEADER_FILE

#include <utility>
#include <random>
#include <vector>

#include "DefTypes.hpp"
#include "utilities/Seeder.hpp"
#include "utilities/Probability.hpp"

namespace MPC_POMDP {
	class Model {
        public:

            /**
             * @brief Basic constructor.
             *
             * This constructor initializes the Model so that all
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
    		Model(size_t s, size_t a, size_t o, double discount = 1.0);

            virtual ~Model() {};

            /**
             * @brief This function replaces the Model termination function with the one provided.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the
             * containers must match the ones provided as arguments
             * (for one dimension: s).
             *
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * Internal values of the container will be converted to bool,
             * so these conversions must be possible.
             *
             * @tparam TER The external terminations container type.
             * @param ter The external terminations container.
             */
            template <typename TER>
            void setTerminationFunction(const TER & ter);

            /**
             * @brief This function replaces the Model termination function with the one provided.
             *
             * The dimensions of the container must match the ones provided
             * as arguments (for one dimensions: S). BE CAREFUL.
             *
             * This function does will throw an std::invalid_argument if the size is not correct.
             *
             * @param ter The external termination container.
             */
            void setTerminationFunction(const std::vector<bool> & ter);

            /**
             * @brief This function replaces the Model violation function with the one provided.
             *
             * The container needs to support data access through
             * operator[]. In addition, the dimensions of the
             * containers must match the ones provided as arguments
             * (for one dimension: s).
             *
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * Internal values of the container will be converted to bool,
             * so these conversions must be possible.
             *
             * @tparam VIO The external violations container type.
             * @param vio The external terminations container.
             */
            template <typename VIO>
            void setViolationFunction(const VIO & vio);

            /**
             * @brief This function replaces the Model violation function with the one provided.
             *
             * The dimensions of the container must match the ones provided
             * as arguments (for one dimensions: S). BE CAREFUL.
             *
             * This function does will throw an std::invalid_argument if the size is not correct.
             *
             * @param vio The external violation container.
             */
            void setViolationFunction(const std::vector<bool> & vio);

            /**
             * @brief This function sets a new discount factor for the Model.
             *
             * @param d The new discount factor for the Model.
             */
            void setDiscount(double d);

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
        	size_t getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
        	size_t getA() const;

            /**
             * @brief This function returns the number of observations the agent could make.
             *
             * @return The total number of observations.
             */
        	size_t getO() const;

            /**
             * @brief This function returns the currently set discount factor.
             *
             * @return The currently set discount factor.
             */
            double getDiscount() const;

            /**
             * @brief This function returns the stored transition probability for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The probability of the specified transition.
             */
        	virtual double getTransitionProbability(size_t s, size_t a, size_t s1) const = 0;

            /**
             * @brief This function returns the stored expected reward for the specified transition.
             * 
             * s1 is actually not used because we only store the expected reward at initial state
             * taking specific actions.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The expected reward of state s taking action a.
             */
        	virtual double getExpectedReward(size_t s, size_t a, size_t s1) const = 0;

            /**
             * @brief This function returns the stored observation probability for the specified state-action pair.
             *
             * @param s1 The final state of the transition.
             * @param a The action performed in the transition.
             * @param o The recorded observation for the transition.
             *
             * @return The probability of the specified observation.
             */
            virtual double getObservationProbability(size_t s1, size_t a, size_t o) const = 0;

            /**
             * @brief This function returns the termination vector.
             *
             * @return The termination vector.
             */
            const std::vector<bool> & getTerminationFunction() const;
            
            /**
             * @brief This function returns the violation vector.
             *
             * @return The violation vector.
             */
            const std::vector<bool> & getViolationFunction() const;

            /**
             * @brief This function checks wheter the given state is the termination.
             *
             * @return True if the state is the termination state.
             */
            bool isTermination(const size_t s) const;

            /**
             * @brief This function checks wheter the given state violates constraints.
             *
             * @return True if the state violate constraints.
             */            
            bool isViolation(const size_t s) const;
            
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
            virtual std::tuple<size_t, size_t, double> propagateSOR(size_t s,size_t a) const = 0;

        protected:
            size_t S, A, O;
            double discount_;
            
            // The termianal states will have value true with others having false
            std::vector<bool> terminations_;
            
            // The constraints violation states have value true, and violation free states having false
            std::vector<bool> violations_;

            mutable RandomEngine rand_;

            /**
            * @brief This function updates trans_end_index according to transitions_
            */
            virtual void updateTransEndIndex() = 0;
	};

    template <typename TER>
    void Model::setTerminationFunction(const TER & ter) {
        terminations_.clear();
        terminations_.resize(S);
        for(size_t s = 0; s < S; ++s) {
            terminations_[s] = ter[s];
        }
    }

    template <typename VIO>
    void Model::setViolationFunction(const VIO & vio) {
        violations_.clear();
        violations_.resize(S);
        for(size_t s = 0; s < S; ++s) {
            violations_[s] = vio[s];
        }
    }
}


#endif