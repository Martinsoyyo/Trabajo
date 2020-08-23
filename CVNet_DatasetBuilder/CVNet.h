#pragma once
#include "cmdlineopt.h"

struct NetworkImpl : torch::nn::SequentialImpl
{
    NetworkImpl() {
        using namespace torch::nn;
        // Layer 1
        push_back(Conv2d(Conv2dOptions(3, 32, 3)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 2
        push_back(Conv2d(Conv2dOptions(32, 32, 3)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 3
        push_back(Conv2d(Conv2dOptions(32, 32, 3)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 4
        push_back(Flatten());
        push_back(Linear(1152, 100));
        push_back(Functional(torch::relu));
        // Layer 5
        push_back(Linear(100, 2));    
        push_back(LogSoftmax(1));
        // Layer 5
        //push_back(Linear(100, 1));
        //push_back(Sigmoid());
    };
};
TORCH_MODULE(Network);

struct BN_Relu_ConvImpl : torch::nn::SequentialImpl {
    BN_Relu_ConvImpl(const size_t& CHANNEL_IN, const size_t& CHANNEL_OUT, const size_t& KERNEL, const float& DROPOUT) {
        using namespace torch::nn;
        push_back(BatchNorm2d(CHANNEL_IN));
        push_back(Functional(torch::relu));
        push_back(Conv2d(Conv2dOptions(CHANNEL_IN, CHANNEL_OUT, KERNEL).padding(1).bias(true)));
        if (DROPOUT > 0.0f) push_back(Dropout(DROPOUT));
    }
};
TORCH_MODULE(BN_Relu_Conv);