#include "IO.hpp"

namespace MPC_POMDP {
    Model parseCassandra(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP(input);

        return Model(S, A, O, T, R, W, TER, VIO, discount);
    }

    SparseModel parseCassandraSparse(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP(input);

        return SparseModel(S, A, O, T, R, W, TER, VIO, discount);
    }

    SparseModel SparseparseCassandraSparse(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP_Sparse(input);

        std::cout << "Finish reading input" << std::endl;

        NoCheck n;

        // SparseModel::RewardMatrix rewards;
        // for ( size_t a = 0; a < A; ++a ) {
        //     for ( size_t s = 0; s < S; ++s )
        //         for ( size_t s1 = 0; s1 < S; ++s1 ) {
        //             const double w = R[a].coeff(s,s1);
        //             const double p = T[a].coeff(s, s1);
        //             if ( checkDifferentSmall(0.0, w) && checkDifferentSmall(0.0, p) ) rewards.coeffRef(s, a) += w * p;
        //     }
        // }
        // rewards.makeCompressed();

        return SparseModel(n, S, A, O, T, R, W, TER, VIO, discount);
    }

}
