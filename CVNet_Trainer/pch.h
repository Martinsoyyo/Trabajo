#pragma once

#include <stdint.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <stdlib.h>
#include <tchar.h>
#include <stdio.h>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define OS_WINDOWS (defined(_WIN32) || defined(_WIN64)...)
#ifdef OS_WINDOWS
	#include <windows.h>
#endif


#include "cxxopts.hpp"