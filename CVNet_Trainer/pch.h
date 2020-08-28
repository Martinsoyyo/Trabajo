#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#ifdef _WIN64
#include <windows.h>
#endif
// NOTA: La otra formulación no funcionaba. Con _WIN64 alcanza ya que en 32 torch no compila


#include "cxxopts.hpp"
