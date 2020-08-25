#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern int  epoch;
	extern int  batch_size;
	extern bool gpu;
	extern bool verbose;
	extern bool overwrite;
	extern int image_size;
	extern float percent_to_train;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};