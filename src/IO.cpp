#include <IO.hpp>

// #include <AIToolbox/POMDP/Utils.hpp>

#include <CassandraParser.hpp>

namespace MPC_POMDP {
    Model<MDP::Model> parseCassandra(std::istream & input) {
        Impl::CassandraParser parser;

        const auto & [S, A, O, T, R, W, discount] = parser.parsePOMDP(input);

        return Model(S, A, O, T, R, W, discount);
    }

}
