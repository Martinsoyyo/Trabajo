#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern float drop_rate;
	extern size_t growth_rate;
	extern std::vector<int> params;
	
	enum TYPE { DENSENET, OTRANET, PIRAMIDAL };
	extern size_t type_net;

	extern size_t  epoch;
	extern size_t  batch_size;
	extern bool cpu;
	extern bool verbose;
	extern size_t batch_norm;
	extern size_t overwrite;
	extern float percent_to_train;
	extern float learning_rate;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};