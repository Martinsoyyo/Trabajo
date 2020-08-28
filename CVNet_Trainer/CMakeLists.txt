# CVNet_Trainer
#
# cmake -DCMAKE_PREFIX_PATH=/home/seba/libtorch ..
# cmake --build . --config Debug


cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CVNet_Trainer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#include_directories(
#    ${CMDLINEOPT_PATH}/include/
#)

add_executable(CVNet_Trainer CVNet_Trainer.cpp Trainer.cpp cmdlineopt.cpp)
# ${CMDLINEOPT_PATH}/include/cmdlineopt.hpp ../common/globals.cpp

# Precompiled header pch.h
target_precompile_headers(CVNet_Trainer PUBLIC pch.h)

# Muestra el tiempo de compilación de cada archivo
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

target_link_libraries(CVNet_Trainer "${TORCH_LIBRARIES}")
set_property(TARGET CVNet_Trainer PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CVNet_Trainer
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch-test>)
endif (MSVC)
