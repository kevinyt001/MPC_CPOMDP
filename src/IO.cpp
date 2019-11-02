#include "IO.hpp"
#include "utilities/CassandraParser.hpp"
#include "Model.hpp"

namespace MPC_POMDP {
    Model parseCassandra(std::istream & input) {
        CassandraParser parser;

        const auto & [S, A, O, T, R, W, TER, VIO, discount] = parser.parsePOMDP(input);

        return Model(S, A, O, T, R, W, TER, VIO, discount);
    }

}
