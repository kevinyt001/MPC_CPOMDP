/**
 * @file parameter_reader.hpp
 * 
 * @copyright Software License Agreement (BSD License)
 * Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick  
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 * 
 * Authors: Zakary Littlefield, Kostas Bekris 
 * 
 */

#ifndef MPC_POMDP_PARAMETER_READER
#define MPC_POMDP_PARAMETER_READER

#include <string>
#include <vector>

namespace params
{
	extern std::string file_name;
	extern int horizon;
	extern int epsilon;
	extern int start_state;
	extern std::vector<int> start_belief;
	extern std::string solver_type;
	extern std::string model_type;
	extern std::string model_scale;
}

void read_parameters(int ac, char* av[]);


#endif