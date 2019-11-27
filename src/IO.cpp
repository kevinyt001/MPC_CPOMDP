#include "IO.hpp"

namespace MPC_POMDP {
    DenseModel* parseCassandra(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP(input);

        return new DenseModel(S, A, O, T, R, W, TER, VIO, discount);
    }

    SparseModel* parseCassandraSparse(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP(input);

        return new SparseModel(S, A, O, T, R, W, TER, VIO, discount);
    }

    SparseModel* parseCassandraLarge(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDPSparse(input);

        NoCheck n;

        return new SparseModel(n, S, A, O, T, R, W, TER, VIO, discount);
    }

}
