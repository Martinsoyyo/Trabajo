#pragma once
#include "cmdlineopt.h"

struct NetworkImpl : torch::nn::SequentialImpl
{
    NetworkImpl(...) {
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


const auto drop_rate = 0.1f;
const uint32_t CHANNEL_IN = 3;
const uint32_t LINEAR = 2;
const uint32_t OUT_CLASSES = 2;
const std::vector<uint32_t> FILTERS = { CHANNEL_IN,32,32,32,32,32,LINEAR };

struct Network3Impl : torch::nn::SequentialImpl
{
    Network3Impl(...) {
        using namespace torch::nn;

        size_t i = 0;
        do {
            push_back(Conv2d(Conv2dOptions(FILTERS[i], FILTERS[i + 1], 3 /*kernel*/).padding(1).bias(false)));
            push_back(BatchNorm2d(FILTERS[i + 1]));
            push_back(MaxPool2d(2));
            push_back(Functional(torch::relu));
            if (drop_rate > 0.0f) push_back(Dropout(drop_rate));

            i++;
        } while (i != FILTERS.size() - 1);

        push_back(Flatten());
        //push_back(Linear(LINEAR, OUT_CLASSES));
        push_back(LogSoftmax(1));
    }
};
TORCH_MODULE(Network3);