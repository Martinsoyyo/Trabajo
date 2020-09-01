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

const uint32_t CHANNEL_IN = 3;
const uint32_t CHANNEL_OUT= 2;

struct OtraNetImpl : torch::nn::SequentialImpl
{
    OtraNetImpl(std::vector<uint32_t>& PARAMS, const float& DROP_RATE) {
        using namespace torch::nn;

        // Armo vector de filtros de tal manera que sea del tipo {CH_in, #params0, #params1,...,CH_out};
        std::vector<uint32_t> FILTERS;
        FILTERS.push_back(CHANNEL_IN);
        for (auto IT : PARAMS) FILTERS.push_back(IT);
        FILTERS.push_back(CHANNEL_OUT);

        size_t i = 0;
        do {
            push_back(Conv2d(Conv2dOptions(FILTERS[i], FILTERS[i + 1], 3 /*kernel*/).padding(1).bias(false)));
            push_back(BatchNorm2d(FILTERS[i + 1]));
            push_back(MaxPool2d(2));
            push_back(Functional(torch::relu));
            if (DROP_RATE > 0.0f) push_back(Dropout(DROP_RATE));

            i++;
        } while (i != FILTERS.size() - 1);

        push_back(Flatten());
        push_back(LogSoftmax(1));
    }
};
TORCH_MODULE(OtraNet);