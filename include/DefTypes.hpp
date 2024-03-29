#ifndef MPC_POMDP_TYPES_HEADER_FILE
#define MPC_POMDP_TYPES_HEADER_FILE

#include <vector>
#include <unordered_map>
#include <random>

#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace MPC_POMDP {
    /**
     * @brief This file defines the types we will use across the implementation of the MPC_POMDP solver.
     */

    // This should have decent properties.
    using RandomEngine = std::mt19937;

    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    using Matrix2D           = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using SparseMatrix2D     = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    using Matrix3D           = std::vector<Matrix2D>;
    using SparseMatrix3D     = std::vector<SparseMatrix2D>;

    using Matrix4D       = boost::multi_array<Matrix2D,       2>;
    using SparseMatrix4D = boost::multi_array<SparseMatrix2D, 2>;

    using Table2D = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using Table3D = std::vector<Table2D>;

    using SparseTable2D = Eigen::SparseMatrix<long, Eigen::RowMajor>;
    using SparseTable3D = std::vector<SparseTable2D>;

    // This is used to store a probability vector (sums to one, every element >= 0, <= 1)
    using ProbabilityVector = Vector;
    using Belief = ProbabilityVector;
    using SparseBelief = Eigen::SparseVector<double>;

    using DumbMatrix2D = boost::multi_array<double, 2>;
    using DumbMatrix3D = boost::multi_array<double, 3>;
    using DumbTable2D  = boost::multi_array<long, 2>;
    using DumbTable3D  = boost::multi_array<long, 3>;

    /**
     * @brief This is used to tag functions that avoid runtime checks.
     */
    inline struct NoCheck {} NO_CHECK;
}

#endif
