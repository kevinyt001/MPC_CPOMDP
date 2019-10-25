#ifndef MPC_POMDP_IO_HEADER_FILE
#define MPC_POMDP_IO_HEADER_FILE

#include <iostream>
#include <iomanip>

#include <Model.hpp>
#include <DefTypes.hpp>

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
    Model parseCassandra(std::istream & input);
}

#endif
