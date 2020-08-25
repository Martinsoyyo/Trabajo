#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern float drop_rate;
	extern uint32_t growth_rate;
	extern std::vector<uint32_t> densenet_params;

	extern int  epoch;
	extern int  batch_size;
	extern bool gpu;
	extern bool verbose;
	extern bool overwrite;
	extern float percent_to_train;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};