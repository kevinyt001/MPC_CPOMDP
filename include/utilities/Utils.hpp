#ifndef MPC_POMDP_UTILS_HEADER_FILE
#define MPC_POMDP_UTILS_HEADER_FILE

#include "DefTypes.hpp"
#include "Model.hpp"

namespace MPC_POMDP {
    /**
     * @brief Creates a new belief reflecting changes after an action 
     * and observation for a particular Model.
     *
     * This function writes directly into the provided Belief pointer. It assumes
     * that the pointer points to a correctly sized Belief. It does a basic nullptr
     * check.
     *
     * This function will not normalize the output, nor is guaranteed
     * to return a non-completely-zero vector.
     *
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template<typename M, typename B>
    void updateBeliefUnnormalized(const M & model, const B & b, 
        const size_t a, const size_t o, B * bRet) {
        
        if (!bRet) return;

        auto & br = *bRet;

        br = model.getObservationFunction(a).col(o).cwiseProduct((b.transpose() * model.getTransitionFunction(a)).transpose());

        // Iteratively update the belief
        /* 
        const size_t S = model.getS();
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            double sum = 0.0;
            for ( size_t s = 0; s < S; ++s )
                sum += model.getTransitionProbability(s,a,s1) * b[s];

            br[s1] = model.getObservationProbability(s1,a,o) * sum;
        }
        */
    }

    /**
     * @brief Creates a new belief reflecting changes after an action 
     * and observation for a particular Model.
     *
     * This function needs to create a new belief since modifying a belief
     * in place is not possible. This is because each cell update for the
     * new belief requires all values from the previous belief.
     *
     * This function will not normalize the output, nor is guaranteed
     * to return a non-completely-zero vector.
     *
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template<typename M, typename B>
    B updateBeliefUnnormalized(const M & model, const B & b, 
        const size_t a, const size_t o) {
        B br(model.getS());
        updateBeliefUnnormalized(model, b, a, o, &br);
        return br;
    }

    /**
     * @brief Creates a new belief reflecting changes after an action 
     * and observation for a particular Model.
     *
     * This function writes directly into the provided Belief pointer. It assumes
     * that the pointer points to a correctly sized Belief. It does a basic nullptr
     * check.
     *
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template<typename M, typename B>
    void updateBelief(const M & model, const B & b, const size_t a, 
        const size_t o, B * bRet) {
        if (!bRet) return;

        updateBeliefUnnormalized(model, b, a, o, bRet);

        auto & br = *bRet;
        br /= br.sum();
    }

    /**
     * @brief Creates a new belief reflecting changes after an action 
     * and observation for a particular Model.
     *
     * This function needs to create a new belief since modifying a belief
     * in place is not possible. This is because each cell update for the
     * new belief requires all values from the previous belief.
     *
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template<typename M, typename B>
    B updateBelief(const M & model, const B & b, const size_t a, 
        const size_t o) {
        B br(model.getS());
        updateBelief(model, b, a, o, &br);
        return br;
    }    

}


#endif