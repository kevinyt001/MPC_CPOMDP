#ifndef MPC_POMDP_MODEL_HEADER_FILE
#define MPC_POMDP_MODEL_HEADER_FILE



namespace MPC_POMDP {
	class Model {
        public:
    		using TransitionMatrix = Matrix3D;
    		using RewardMatrix = Matrix2D;
    		using ObservationMatrix = Matrix3D;
    		using Belief = ProbabilityVector;

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
            Model(const Model& model);

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
             * @param s  The number of states of the world.
             * @param a  The number of actions available to the agent.
             * @param o  The number of observations available to the agent.
             * @param t  The external transitions container.
             * @param r  The external rewards container.
             * @param om The external observations container.
             * @param d  The discount factor for the POMDP.
             */
            template <typename T, typename R, typename OM>
            Model(size_t s, size_t a, size_t o, const T & t, const R & r, 
            	const OM & om, double d = 1.0);

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
            Model(NoCheck, size_t s, size_t a, size_t o, TransitionMatrix && t, 
            	RewardMatrix && r, ObservationMatrix && om, double d);

            virtual ~Model();

            /**
             * @brief This function replaces the Model transition function with the one provided.
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
             * @brief This function sets the transition function using a Eigen dense matrices.
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
             * @param r The external rewards container.
             */
            void setObservationFunction(const ObservationMatrix & om);

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
             * @return The rewards matrix.
             */
            const TransitionMatrix & getTransitionFunction() const;

            /**
             * @brief This function returns the transition function for a given action.
             *
             * @param a The action requested.
             *
             * @return The transition function for the input action.
             */
            const Matrix2D & getTransitionFunction(size_t a) const;

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
            const Matrix2D & getObservationFunction(size_t a) const;

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
            std::tuple<size_t,size_t, double> propagateSOR(size_t s,size_t a) const;

        private:
            size_t S, A, O;
            double discount_;
		
            TransitionMatrix transitions_;
            RewardMatrix rewards_; //Contain the expected reward at state s with action a
            ObservationMatrix obseravtions_; //Observation Matrix for each action is a probability distribution
            // Belief beliefs_;

            mutable RandomEngine rand_;
	};
}


#endif