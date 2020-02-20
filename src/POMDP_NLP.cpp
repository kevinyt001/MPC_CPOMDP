#include "POMDP_NLP.hpp"

#include <cassert>
#include <iostream>

#include "SparseModel.hpp"
#include "DefTypes.hpp"

using namespace Ipopt;

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace MPC_POMDP {

   // constructor
   // POMDP_NLP::POMDP_NLP():
   //    m_(SparseModel(1,1,1)), b_(Belief(1)){}

   POMDP_NLP::POMDP_NLP(const SparseModel& m_in, const Belief& b_in, int horizon_in, double  epsilon_in) :
      m_(m_in), b_(b_in), horizon_(horizon_in), epsilon_(epsilon_in) {}


   // destructor
   POMDP_NLP::~POMDP_NLP()
   { }

   // [TNLP_get_nlp_info]
   // returns the size of the problem
   bool POMDP_NLP::get_nlp_info(
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
   bool POMDP_NLP::get_bounds_info(
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

      // the variables have lower bounds of 0
      for( Index i = 0; i < n; i++ )
      {
         x_l[i] = 0.0;
      }

      // the variables have upper bounds of 1
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
   bool POMDP_NLP::get_starting_point(
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
   bool POMDP_NLP::eval_f(
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
   bool POMDP_NLP::eval_grad_f(
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
   bool POMDP_NLP::eval_g(
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
         Vector temp_g(m_.getA());
         for (size_t j = 0; j < m_.getA(); ++j) 
             temp_g(j) = x[i*m_.getA()+j];

         // Update belief 
         //Update together
         predict_belief.setZero();
         for(size_t j = 0; j < m_.getA(); ++j) {
           predict_belief.noalias() += Eigen::VectorXd(temp_g(j) * m_.getTransitionFunction(j).transpose() * temp);
         }

         for (size_t j = 0; j < m_.getS(); ++j) {
           if(checkDifferentSmall(predict_belief(j), 0.0) && m_.isViolation(j)) {
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
   bool POMDP_NLP::eval_jac_g(
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
   bool POMDP_NLP::eval_h(
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
      std::cout << "Start evaluating h" << std::endl;

      assert(n == 4);
      assert(m == 2);

      if( values == NULL )
      {
         // return the structure. This is a symmetric matrix, fill the lower left
         // triangle only.

         // the hessian for this problem is actually dense
         // Index idx = 0;
         // for( Index row = 0; row < 4; row++ )
         // {
         //    for( Index col = 0; col <= row; col++ )
         //    {
         //       iRow[idx] = row;
         //       jCol[idx] = col;
         //       idx++;
         //    }
         // }

         // assert(idx == nele_hess);
      }
      else
      {
         // return the values. This is a symmetric matrix, fill the lower left
         // triangle only

         // fill the objective portion
         // values[0] = obj_factor * (2 * x[3]); // 0,0

         // values[1] = obj_factor * (x[3]);     // 1,0
         // values[2] = 0.;                      // 1,1

         // values[3] = obj_factor * (x[3]);     // 2,0
         // values[4] = 0.;                      // 2,1
         // values[5] = 0.;                      // 2,2

         // values[6] = obj_factor * (2 * x[0] + x[1] + x[2]); // 3,0
         // values[7] = obj_factor * (x[0]);                   // 3,1
         // values[8] = obj_factor * (x[0]);                   // 3,2
         // values[9] = 0.;                                    // 3,3

         // // add the portion for the first constraint
         // values[1] += lambda[0] * (x[2] * x[3]); // 1,0

         // values[3] += lambda[0] * (x[1] * x[3]); // 2,0
         // values[4] += lambda[0] * (x[0] * x[3]); // 2,1

         // values[6] += lambda[0] * (x[1] * x[2]); // 3,0
         // values[7] += lambda[0] * (x[0] * x[2]); // 3,1
         // values[8] += lambda[0] * (x[0] * x[1]); // 3,2

         // // add the portion for the second constraint
         // values[0] += lambda[1] * 2; // 0,0

         // values[2] += lambda[1] * 2; // 1,1

         // values[5] += lambda[1] * 2; // 2,2

         // values[9] += lambda[1] * 2; // 3,3
      }

      return false;
   }
   // [TNLP_eval_h]

   // [TNLP_finalize_solution]
   void POMDP_NLP::finalize_solution(
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

      std::ofstream ofs;
      ofs.open("results.ipopt", std::ofstream::out);
      for( Index i = 0; i < n; i++ )
      {
         ofs << x[i] << std::endl;
      }
      
      // Number temp[4];
      // eval_g(n, x, m, temp);
      // std::cout << std::endl << "Final value of the constraints:" << std::endl;
      // for( Index i = 0; i < m; i++ )
      // {
      //    std::cout << "g(" << i << ") = " << (double) temp[i] << std::endl;
      // }

      // std::cout << "Belief: " << std::endl;
      // for(size_t i = 0; i < m_.getS(); i ++) {
      //     if(checkDifferentSmall(b_(i), 0.0))
      //         std::cout << "S: " << i << " B: " << b_(i) << std::endl;
      // }

      // // For this example, we write the solution to the console
      // std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
      // for( Index i = 0; i < n; i++ )
      // {
      //    std::cout << "x[" << i << "] = " << x[i] << std::endl;
      // }

      // for( Index i = 0; i < n; i++ )
      // {
      //    std::cout << x[i] << " ";
      // }
      // std::cout << std::endl;

      // std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
      // for( Index i = 0; i < n; i++ )
      // {
      //    std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
      // }
      // for( Index i = 0; i < n; i++ )
      // {
      //    std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
      // }

      // std::cout << std::endl << std::endl << "Objective value" << std::endl;
      // std::cout << "f(x*) = " << obj_value << std::endl;

      // std::cout << std::endl << "Final value of the constraints:" << std::endl;
      // for( Index i = 0; i < m; i++ )
      // {
      //    std::cout << "g(" << i << ") = " << g[i] << std::endl;
      // }
   }
   // [TNLP_finalize_solution]

   // For debug purpose only
   bool POMDP_NLP::eval_g(
      Index         n,
      const Number* x,
      Index         m,
      Number*       g
   )
   {
      // std::cout << "Start evaluating g" << std::endl;

      assert(n == horizon_*m_.getA());
      assert(m == 1 + horizon_);

      Belief predict_belief = b_;
      g[0] = 0.0;
      for (int i = 0; i < horizon_; ++i) {
         Belief temp = predict_belief;
         Vector temp_g(m_.getA());
         for (size_t j = 0; j < m_.getA(); ++j) 
             temp_g(j) = x[i*m_.getA()+j];

         // Update belief 
         //Update together
         predict_belief.setZero();
         for(size_t j = 0; j < m_.getA(); ++j) {
           predict_belief.noalias() += Eigen::VectorXd(temp_g(j) * m_.getTransitionFunction(j).transpose() * temp);
         }

         std::cout << "Debug constraint calculation" << std::endl;
         std::cout << predict_belief(6667) << std::endl;
         std::cout << m_.isViolation(6667) << std::endl;
         std::cout << std::endl;

         for (size_t j = 0; j < m_.getS(); ++j) {
           if(checkDifferentSmall(predict_belief(j), 0.0) && m_.isViolation(j)) {
               g[0] += predict_belief(j);
               std::cout << "values of g[0]" << g[0] << std::endl;
               predict_belief(j) = 0;
           }
         }
      }

      std::cout << "values " << g[0] << std::endl;

      for (int i = 1; i < m; ++i) {
         g[i] = 0.0;
         for (size_t j = 0; j < m_.getA(); ++j) {
            g[i] += x[(i-1)*m_.getA() + j];
         }
      }

      std::cout << "values " << g[0] << std::endl;

      // std::cout << "Finish evaluating g" << std::endl;

      return true;
   }
}