#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef _WIN64
#include <windows.h>
#endif
// NOTA: La otra formulación no funcionaba. Con _WIN64 alcanza ya que en 32 torch no compila


#include "cxxopts.hpp"
