
#include <iostream>
#include <iomanip> 		/* setprecision */
#include <fstream>
#include <stdlib.h>     /* srand, rand, abs*/
#include <time.h>       /* time */

#include <nlopt.hpp>

#include "models/DenseModel.hpp"
#include "utilities/CassandraParser.hpp"
#include "IO.hpp"
#include "utilities/Core.hpp"

int main() {
	
	std::ifstream ifs;
	ifs.open("../input/overtake.POMDP", std::ifstream::in);

	// MPC_POMDP::DenseModel* overtake = MPC_POMDP::parseCassandra(ifs);

	// std::ofstream ofs;
	// ofs.open("test_input.POMDP", std::ofstream::out);

	// ofs << std::endl;

	// ofs << "states: " << overtake.getS() << std::endl;
	// ofs << "actions: " << overtake.getA() << std::endl;
	// ofs << "observations: " << overtake.getO() << std::endl;

	// ofs << std::endl;
	
	// // Print Transition	
	// for (size_t a = 0; a < overtake.getA(); ++a) {
	// 	for (size_t s = 0; s < overtake.getS(); ++s) {
	// 		for (size_t s1 = 0; s1 < overtake.getS(); ++s1) {
	// 			if(overtake.getTransitionProbability(s, a, s1) > MPC_POMDP::equalToleranceGeneral) {
	// 				ofs << "T : " << a << " : " << s << " : " << s1 << " " 
	// 				<< std::fixed << std::setprecision(2) 
	// 				<< overtake.getTransitionProbability(s, a, s1) << std::endl;
	// 			}
	// 		}
	// 	}
	// }

	// ofs << std::endl;

	// // Print Observation
	// srand(time(NULL));
	// for (size_t s1 = 0; s1 < overtake.getS(); ++s1) {
	// 	size_t a = rand() % overtake.getA();
	// 	for (size_t o = 0; o < overtake.getO(); ++o) {
	// 		if(overtake.getObservationProbability(s1, a, o) > MPC_POMDP::equalToleranceGeneral) {
	// 			ofs << "O : * : " << s1 << " : " << o << " " 
	// 			<< std::fixed << std::setprecision(2) 
	// 			<< overtake.getObservationProbability(s1, a, o) << std::endl;
	// 		}
	// 	}
	// }

	// ofs << std::endl;

	// // Print Rewards
	// for (size_t a = 0; a < overtake.getA(); ++a) {
	// 	for (size_t s = 0; s < overtake.getS(); ++s) {
	// 		if(abs(overtake.getExpectedReward(s, a, 0)) > MPC_POMDP::equalToleranceGeneral) {
	// 			ofs << "R : " << a << " : " << s << " : * : * " 
	// 			<< std::fixed << std::setprecision(2) 
	// 			<< overtake.getExpectedReward(s, a, 0) << std::endl;
	// 		}
	// 	}
	// }

	// ofs << std::endl;

	// // Print Terminations
	// for (size_t s = 0; s < overtake.getS(); ++s) {
	// 	if (overtake.isTermination(s)) {
	// 		ofs << "E : " << s << std::endl;
	// 	}
	// }

	// ofs << std::endl;

	// // Print Violations
	// for (size_t s = 0; s < overtake.getS(); ++s) {
	// 	if (overtake.isViolation(s)) {
	// 		ofs << "V : " << s << std::endl;
	// 	}
	// }

	// ofs.close();

	// Check trans_end_index_ is properly set up

	return 0;
}