#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern float drop_rate;
	extern uint32_t growth_rate;
	extern std::vector<uint32_t> params;
	
	enum TYPE { DENSENET, OTRANET };
	extern int type_net;

	extern int  epoch;
	extern int  batch_size;
	extern bool cpu;
	extern bool verbose;
	extern bool overwrite;
	extern float percent_to_train;
	extern float learning_rate;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};