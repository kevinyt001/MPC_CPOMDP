#include "utilities/parameter_reader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

namespace params
{
	std::string file_name;
	int horizon;
	int epsilon;
	int start_state;
	std::vector<int> start_belief;
	std::string solver_type;
	std::string model_type;
	std::string model_scale;
}

#include <boost/program_options.hpp>

namespace po = boost::program_options;

void read_parameters(int ac, char* av[])
{

	std::string config_file_name; 
	po::options_description opt_desc("Options"); 
	opt_desc.add_options() 
	("help","Print available options.") ("config", po::value< std::string >( &config_file_name )->default_value("../input/default.cfg"), 
	"The name of a file to read for options (default is ../input/default.cfg). Command-line options" 
	" override the ones in the config file. A config file may contain lines with syntax" 
	"\n'long_option_name = value'\nand comment lines that begin with '#'." )
	("file_name",po::value<std::string>(&params::file_name),"Location and name of the file containing the model information in cassandra format.") 
	("horizon",po::value<int>(&params::horizon),"Horizon of MPC POMDP to solve the optimal control distribution (horizon >= 1).") 
	("epsilon",po::value<int>(&params::epsilon),"Chance constraints rate (0 <= epsilon <= 1).") 
	("start_state",po::value<int>(&params::start_state),"Start state index of the system.") 
	("start_belief",po::value<std::string>(),"The start belief index of the system. Belief is evenly distributed. Input is in the format of \"0 1\"") 
	("solver_type",po::value<std::string>(&params::solver_type),"Type of solver to use (ipopt or nlopt).") 
	("model_type",po::value<std::string>(&params::model_type),"Type of model to use (dense or sparse).") 
	("model_scale",po::value<std::string>(&params::model_scale),"Scale of the model (small or large).") 
	;

    po::variables_map varmap;
	po::store(po::parse_command_line(ac,av,opt_desc),varmap); 
	po::notify( varmap ); 

	if (varmap.count("help")) 
	{
    	std::cout << opt_desc << "\n";
	    exit(0);
	}

	if( varmap.count("config") ) 
	{ 
	  std::cout << "Loading options from " << config_file_name; 
	  std::cout.flush(); 
	  std::ifstream ifs( config_file_name.c_str() ); 
	  if( !ifs.is_open() ) 
	   std::cout << "no such file." << std::endl; 
	  else 
	  { 
	   po::store( po::parse_config_file( ifs, opt_desc ), varmap ); 
	   po::notify( varmap ); 
	   std::cout << " done." << std::endl; 
	  } 
	} 

	std::vector<double> belief;
	if (varmap.count("start_belief")) 
	{
		std::stringstream stream(varmap["start_belief"].as<std::string>());
		double n; while(stream >> n) {belief.push_back(n);}
		// belief = varmap["start_belief"].as<std::vector<double> >();
		params::start_belief.resize(belief.size());
		for(unsigned i=0;i<belief.size();i++)
		{
			params::start_belief[i] = belief[i];
		}
	}

}