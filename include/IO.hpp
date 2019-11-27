#ifndef MPC_POMDP_IO_HEADER_FILE
#define MPC_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include "models/DenseModel.hpp"
#include "models/SparseModel.hpp"
#include "DefTypes.hpp"
#include "utilities/CassandraParser.hpp"

namespace MPC_POMDP {
	/**
     * @brief This function parses a POMDP from a Cassandra formatted stream.
     *
     * This function may throw std::runtime_errors depending on whether the
     * input is correctly formed or not.
     *
     * @param input The input stream.
     *
     * @return The parsed model.
     */
    DenseModel* parseCassandra(std::istream & input);

    /**
     * @brief This function parses a POMDP from a Cassandra formatted stream.
     *
     * This function may throw std::runtime_errors depending on whether the
     * input is correctly formed or not.
     *
     * @param input The input stream.
     *
     * @return The parsed sparse model.
     */
    SparseModel* parseCassandraSparse(std::istream & input);

    /**
     * @brief This function parses a POMDP from a Cassandra formatted stream
     * using Sparse matrices directly.
     *
     * This function is used to parse extremely large models (with states over
     * ten thousand).
     *
     * Note that in order to properly set up the model, you need to completely 
     * specify transitions before rewards in the input stream. This function 
     * will not check model validities.
     *
     * This function may throw std::runtime_errors depending on whether the
     * input is correctly formed or not.
     *
     * @param input The input stream.
     *
     * @return The parsed sparse model.
     */
    SparseModel* parseCassandraLarge(std::istream & input);

}

#endif
