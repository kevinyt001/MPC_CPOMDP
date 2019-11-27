#ifndef MPC_POMDP_POMDP_NLP_HEADER_FILE
#define MPC_POMDP_POMDP_NLP_HEADER_FILE

#include "IpTNLP.hpp"
#include "models/SparseModel.hpp"
#include "models/DenseModel.hpp"
#include "DefTypes.hpp"

#include <iostream>
#include <cassert>

using namespace Ipopt;

namespace MPC_POMDP {
	template<typename M, typename B>
	class POMDP_NLP: public TNLP 
	{
	public:
		/** Default constructor */
		// POMDP_NLP();

		POMDP_NLP(const M& m_in, const B& b_in, int horizon_in, double epsilon_in);

		/** Default destructor */
		virtual ~POMDP_NLP();

		/**@name Overloaded from TNLP */
		//@{
		/** Method to return some info about the NLP */
	    virtual bool get_nlp_info(
	       Index&          n,
	       Index&          m,
	       Index&          nnz_jac_g,
	       Index&          nnz_h_lag,
	       IndexStyleEnum& index_style
	    );

	    /** Method to return the bounds for my problem */
	    virtual bool get_bounds_info(
	       Index   n,
	       Number* x_l,
	       Number* x_u,
	       Index   m,
	       Number* g_l,
	       Number* g_u
	    );

	    /** Method to return the starting point for the algorithm */
	    virtual bool get_starting_point(
	       Index   n,
	       bool    init_x,
	       Number* x,
	       bool    init_z,
	       Number* z_L,
	       Number* z_U,
	       Index   m,
	       bool    init_lambda,
	       Number* lambda
	    );

	    /** Method to return the objective value */
	    virtual bool eval_f(
	       Index         n,
	       const Number* x,
	       bool          new_x,
	       Number&       obj_value
	    );

	    /** Method to return the gradient of the objective */
	    virtual bool eval_grad_f(
	       Index         n,
	       const Number* x,
	       bool          new_x,
	       Number*       grad_f
	    );

	    /** Method to return the constraint residuals */
	    virtual bool eval_g(
	       Index         n,
	       const Number* x,
	       bool          new_x,
	       Index         m,
	       Number*       g
	    );

	    /** Method to return:
	     *   1) The structure of the jacobian (if "values" is NULL)
	     *   2) The values of the jacobian (if "values" is not NULL)
	     */
	    virtual bool eval_jac_g(
	       Index         n,
	       const Number* x,
	       bool          new_x,
	       Index         m,
	       Index         nele_jac,
	       Index*        iRow,
	       Index*        jCol,
	       Number*       values
	    );

	    /** Method to return:
	     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
	     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
	     */
	    virtual bool eval_h(
	       Index         n,
	       const Number* x,
	       bool          new_x,
	       Number        obj_factor,
	       Index         m,
	       const Number* lambda,
	       bool          new_lambda,
	       Index         nele_hess,
	       Index*        iRow,
	       Index*        jCol,
	       Number*       values
	    );

	    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
	    virtual void finalize_solution(
	       SolverReturn               status,
	       Index                      n,
	       const Number*              x,
	       const Number*              z_L,
	       const Number*              z_U,
	       Index                      m,
	       const Number*              g,
	       const Number*              lambda,
	       Number                     obj_value,
	       const IpoptData*           ip_data,
	       IpoptCalculatedQuantities* ip_cq
	    );
	    //@}

	private:
		const M& m_;
		const B& b_;
		int horizon_;
		double epsilon_;

	    /**@name Methods to block default compiler methods.
	     *
	     * The compiler automatically generates the following three methods.
	     *  Since the default compiler implementation is generally not what
	     *  you want (for all but the most simple classes), we usually
	     *  put the declarations of these methods in the private section
	     *  and never implement them. This prevents the compiler from
	     *  implementing an incorrect "default" behavior without us
	     *  knowing. (See Scott Meyers book, "Effective C++")
	     */
	    //@{
	    POMDP_NLP(
	       const POMDP_NLP&
	    );

	    POMDP_NLP& operator=(
	       const POMDP_NLP&
	    );
	    //@}
	};

	template<typename M, typename B>
	POMDP_NLP<M, B>::POMDP_NLP(const M& m_in, const B& b_in, int horizon_in, double  epsilon_in) :
	  m_(m_in), b_(b_in), horizon_(horizon_in), epsilon_(epsilon_in) {}

	// destructor
	template<typename M, typename B>
	POMDP_NLP<M, B>::~POMDP_NLP()
	{ }

	// [TNLP_get_nlp_info]
	// returns the size of the problem
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::get_nlp_info(
	  Index&          n,
	  Index&          m,
	  Index&          nnz_jac_g,
	  Index&          nnz_h_lag,
	  IndexStyleEnum& index_style
	)
	{
	  std::cout << "Start getting NLP info" << std::endl;
	  // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
	  n = horizon_ * m_.getA();

	  // one equality constraint and one inequality constraint
	  m = 1 + horizon_;

	  // in this example the jacobian is dense and contains 8 nonzeros
	  nnz_jac_g = n + n;

	  // the Hessian is also dense and has 16 total nonzeros, but we
	  // only need the lower left corner (since it is symmetric)
	  nnz_h_lag = n*n;

	  // use the C style indexing (0-based)
	  index_style = TNLP::C_STYLE;

	  std::cout << "Finish getting NLP info" << std::endl;

	  return true;
	}
	// [TNLP_get_nlp_info]

	// [TNLP_get_bounds_info]
	// returns the variable bounds
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::get_bounds_info(
	  Index   n,
	  Number* x_l,
	  Number* x_u,
	  Index   m,
	  Number* g_l,
	  Number* g_u
	)
	{
	  std::cout << "Start getting NLP bound info" << std::endl;
	  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
	  // If desired, we could assert to make sure they are what we think they are.
	  assert(n == horizon_ * m_.getA());
	  assert(m == 1 + horizon_);

	  // the variables have lower bounds of 1
	  for( Index i = 0; i < n; i++ )
	  {
	     x_l[i] = 0.0;
	  }

	  // the variables have upper bounds of 5
	  for( Index i = 0; i < n; i++ )
	  {
	     x_u[i] = 1.0;
	  }

	  // the first constraint g1 has a lower bound of 25
	  g_l[0] = 0.0;
	  // the first constraint g1 has NO upper bound, here we set it to 2e19.
	  // Ipopt interprets any number greater than nlp_upper_bound_inf as
	  // infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
	  // is 1e19 and can be changed through ipopt options.
	  g_u[0] = epsilon_;

	  // the second constraint g2 is an equality constraint, so we set the
	  // upper and lower bound to the same value
	  for(Index i = 1; i < m; i++) {
	     g_l[i] = g_u[i] = 1.0;
	  }

	  std::cout << "Finisth getting NLP bound info" << std::endl;

	  return true;
	}
	// [TNLP_get_bounds_info]

	// [TNLP_get_starting_point]
	// returns the initial point for the problem
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::get_starting_point(
	  Index   n,
	  bool    init_x,
	  Number* x,
	  bool    init_z,
	  Number* z_L,
	  Number* z_U,
	  Index   m,
	  bool    init_lambda,
	  Number* lambda
	)
	{
	  std::cout << "Start getting NLP starting point" << std::endl;
	  // Here, we assume we only have starting values for x, if you code
	  // your own NLP, you can provide starting values for the dual variables
	  // if you wish
	  assert(init_x == true);
	  assert(init_z == false);
	  assert(init_lambda == false);

	  // initialize to the given starting point
	  for(Index i = 0; i < n; ++i) {
	     x[i] = 1.0/(double) n;
	  }

	  std::cout << "Finish getting NLP starting point" << std::endl;

	  return true;
	}
	// [TNLP_get_starting_point]

	// [TNLP_eval_f]
	// returns the value of the objective function
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::eval_f(
	  Index         n,
	  const Number* x,
	  bool          new_x,
	  Number&       obj_value
	)
	{
	  // std::cout << "Start evaluating f" << std::endl;
	  assert(n == horizon_*m_.getA());
	  
	  obj_value = 0;
	  Belief predict_belief = b_;
	  // clock_t t = clock();
	  for (int i = 0; i < horizon_; ++i) {
	     Vector g(m_.getA());
	     for (size_t j = 0; j < m_.getA(); ++j)
	         g(j) = x[i*m_.getA()+j];

	     // belief.transpose: 1*S; rewards_: S*A; g(gamma): A*1
	     obj_value += predict_belief.transpose() * m_.getRewardFunction() * g;

	     // Update belief
	     if(i == horizon_-1) break;
	     Belief temp = predict_belief;

	     //Update together
	     predict_belief.setZero();
	     for(size_t j = 0; j < m_.getA(); ++j) {
	       predict_belief.noalias() += Eigen::VectorXd(g(j) * m_.getTransitionFunction(j).transpose() * temp);
	     }
	  }

	  // std::cout << "Finish evaluating f" << std::endl;

	  return true;
	}
	// [TNLP_eval_f]

	// [TNLP_eval_grad_f]
	// return the gradient of the objective function grad_{x} f(x)
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::eval_grad_f(
	  Index         n,
	  const Number* x,
	  bool          new_x,
	  Number*       grad_f
	)
	{
	  // std::cout << "Start evaluating grad f" << std::endl;
	  assert(n == horizon_*m_.getA());

	  Number* temp_x = new Number[horizon_*m_.getA()];
	  for(Index i = 0; i < n; ++i) {
	     temp_x[i] = x[i];
	  }

	  for(Index i = 0; i < n; ++i) {
	      temp_x[i] = std::min(x[i] + std::max(x[i]*0.05, 0.01), 1.0);
	      // temp_gamma[i] = gamma[i] + std::max(gamma[i]*0.05, 0.01);
	      Number adj1 = temp_x[i] - x[i];
	      Number temp1 = 0;
	      eval_f(n, temp_x, true, temp1);
	      temp_x[i] = std::max(x[i] - std::max(x[i]*0.05, 0.01), 0.0);
	      // temp_gamma[i] = gamma[i] - std::max(gamma[i]*0.05, 0.01);
	      Number adj2 = temp_x[i] - x[i];
	      Number temp2 = 0;
	      eval_f(n, temp_x, true, temp2);

	      grad_f[i] = (temp1 - temp2) / (adj1 - adj2);

	      temp_x[i] = x[i];
	  }
	  delete temp_x;

	  // std::cout << "Finish evaluating grad f" << std::endl;

	  return true;
	}
	// [TNLP_eval_grad_f]

	// [TNLP_eval_g]
	// return the value of the constraints: g(x)
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::eval_g(
	  Index         n,
	  const Number* x,
	  bool          new_x,
	  Index         m,
	  Number*       g
	)
	{
	  // std::cout << "Start evaluating g" << std::endl;

	  assert(n == horizon_*m_.getA());
	  assert(m == 1 + horizon_);

	  Belief predict_belief = b_;
	  g[0] = 0;
	  for (int i = 0; i < horizon_; ++i) {
	     Belief temp = predict_belief;
	     Vector g(m_.getA());
	     for (size_t j = 0; j < m_.getA(); ++j) 
	         g(j) = x[i*m_.getA()+j];

	     // Update belief 
	     //Update together
	     predict_belief.setZero();
	     for(size_t j = 0; j < m_.getA(); ++j) {
	       predict_belief.noalias() += Eigen::VectorXd(g(j) * m_.getTransitionFunction(j).transpose() * temp);
	     }

	     for (size_t j = 0; j < m_.getS(); ++j) {
	       if(checkDifferentSmall(predict_belief(j), 0) && m_.isViolation(j)) {
	           g[0] += predict_belief(j);
	           predict_belief(j) = 0;
	       }
	     }
	  }

	  for (Index i = 1; i < m; ++i) {
	     g[i] = 0;
	     for (size_t j = 0; j < m_.getA(); ++j) {
	        g[i] += x[(i-1)*m_.getA() + j];
	     }
	  }

	  // std::cout << "Finish evaluating g" << std::endl;

	  return true;
	}
	// [TNLP_eval_g]

	// [TNLP_eval_jac_g]
	// return the structure or values of the Jacobian
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::eval_jac_g(
	  Index         n,
	  const Number* x,
	  bool          new_x,
	  Index         m,
	  Index         nele_jac,
	  Index*        iRow,
	  Index*        jCol,
	  Number*       values
	)
	{
	  // std::cout << "Start evaluating jac g" << std::endl;

	  assert(n == horizon_*m_.getA());
	  assert(m == 1 + horizon_);

	  if( values == NULL )
	  {
	     /*
	     return the structure of the Jacobian
	     iRow[idx] = row, iCol[idx] = col means values[idx] stands for 
	     (row, col) element in the Jacobian matrix
	     */
	     Index idx = 0;
	     Index row = 0;
	     for(Index col = 0; col < n; col++) {
	        iRow[idx] = row;
	        jCol[idx] = col;
	        idx++;
	     }

	     for(row = 1; row < m; row++ )
	     {
	        for(Index col = 0; col < m_.getA(); col++) {
	           iRow[idx] = row;
	           jCol[idx] = (row-1)*m_.getA() + col;
	           idx++;
	        }
	     }

	     // for(Index i = 0; i < idx; ++i) {
	     //    std::cout<< "idx: " << i << "(" << iRow[i] << "," << jCol[i] << ")" << std::endl;
	     // }

	     // assert(false);

	  }
	  else
	  {
	     // return the values of the Jacobian of the constraints
	     Number* temp_x = new Number[horizon_*m_.getA()];
	     for(Index i = 0; i < n; ++i) {
	        temp_x[i] = x[i];
	     }
	     Number* temp_g1 = new Number[m];
	     Number* temp_g2 = new Number[m];

	     for(Index i = 0; i < n; ++i) {
	         temp_x[i] = std::min(x[i] + std::max(x[i]*0.05, 0.01), 1.0);
	         // temp_gamma[i] = gamma[i] + std::max(gamma[i]*0.05, 0.01);
	         Number adj1 = temp_x[i] - x[i];
	         eval_g(n, temp_x, true, m, temp_g1);
	         temp_x[i] = std::max(x[i] - std::max(x[i]*0.05, 0.01), 0.0);
	         // temp_gamma[i] = gamma[i] - std::max(gamma[i]*0.05, 0.01);
	         Number adj2 = temp_x[i] - x[i];
	         eval_g(n, temp_x, true, m, temp_g2);

	         values[i] = (temp_g1[0] - temp_g2[0]) / (adj1 - adj2);

	         temp_x[i] = x[i];
	     }
	     delete temp_x; delete temp_g1; delete temp_g2;

	     for(Index i = n; i < nele_jac; ++i) 
	        values[i] = 1;
	  }

	  // std::cout << "Finish evaluating jac g" << std::endl;

	  return true;
	}
	// [TNLP_eval_jac_g]

	// [TNLP_eval_h]
	//return the structure or values of the Hessian
	template<typename M, typename B>
	bool POMDP_NLP<M, B>::eval_h(
	  Index         n,
	  const Number* x,
	  bool          new_x,
	  Number        obj_factor,
	  Index         m,
	  const Number* lambda,
	  bool          new_lambda,
	  Index         nele_hess,
	  Index*        iRow,
	  Index*        jCol,
	  Number*       values
	)
	{
	  return false;
	}
	// [TNLP_eval_h]

	// [TNLP_finalize_solution]
	template<typename M, typename B>
	void POMDP_NLP<M, B>::finalize_solution(
	  SolverReturn               status,
	  Index                      n,
	  const Number*              x,
	  const Number*              z_L,
	  const Number*              z_U,
	  Index                      m,
	  const Number*              g,
	  const Number*              lambda,
	  Number                     obj_value,
	  const IpoptData*           ip_data,
	  IpoptCalculatedQuantities* ip_cq
	)
	{
	  // here is where we would store the solution to variables, or write to a file, etc
	  // so we could use the solution.

	  // For this example, we write the solution to the console
	  std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
	  for( Index i = 0; i < n; i++ )
	  {
	     std::cout << "x[" << i << "] = " << x[i] << std::endl;
	  }

	  for( Index i = 0; i < n; i++ )
	  {
	     std::cout << x[i] << " ";
	  }
	  std::cout << std::endl;

	  std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
	  for( Index i = 0; i < n; i++ )
	  {
	     std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
	  }
	  for( Index i = 0; i < n; i++ )
	  {
	     std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
	  }

	  std::cout << std::endl << std::endl << "Objective value" << std::endl;
	  std::cout << "f(x*) = " << obj_value << std::endl;

	  std::cout << std::endl << "Final value of the constraints:" << std::endl;
	  for( Index i = 0; i < m; i++ )
	  {
	     std::cout << "g(" << i << ") = " << g[i] << std::endl;
	  }
	}
	// [TNLP_finalize_solution]

}

#endif