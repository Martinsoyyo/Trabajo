#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern float drop_rate;
	extern uint32_t growth_rate;
	extern std::vector<int> params;
	
	enum TYPE { DENSENET, OTRANET, PIRAMIDAL };
	extern int type_net;

	extern int  epoch;
	extern int  batch_size;
	extern bool cpu;
	extern bool verbose;
	extern bool batch_norm;
	extern int overwrite;
	extern float percent_to_train;
	extern float learning_rate;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};