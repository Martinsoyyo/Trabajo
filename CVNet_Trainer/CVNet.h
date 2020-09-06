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


// USO: ! time ./CVNet_Trainer --path /content --prefix 64x64 --epoch 80 --batch 128 --use otranet --params 32,32 --drop_rate 0.25 --learning_rate 0.0025
const uint32_t CHANNEL_IN = 3;
const uint32_t CHANNEL_OUT= 2;

struct OtraNetImpl : torch::nn::SequentialImpl
{
    torch::nn::Sequential features, classifier;

    OtraNetImpl(std::vector<int>& PARAMS, const float& DROP_RATE,  const bool& BATCH_NORM) {
        using namespace torch::nn;

        size_t channel = CHANNEL_IN;
        for (const auto& V : PARAMS)
        {
            if (V <= -1) features->push_back(MaxPool2d(2));
            else {
                features->push_back(Conv2d(Conv2dOptions(channel, V, 3 /*kernel*/).padding(1).bias(false)));
                if (BATCH_NORM) features->push_back(BatchNorm2d(V));
                if (!DROP_RATE) features->push_back(Dropout(DROP_RATE));
                features->push_back(Functional(torch::relu));
                channel = V;
            }
        };

        classifier = torch::nn::Sequential(
            Linear(channel*4*4, 128),
            Functional(torch::relu),
            Dropout(DROP_RATE),
            Linear(128, 128),
            Functional(torch::relu),
            Dropout(DROP_RATE),
            Linear(128, 2 /* OUT_CLASES */),
            LogSoftmax(1)
        );

        register_module("features", this->features);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = x.view({ x.size(0), -1 });

        // x = torch::adaptive_avg_pool2d(x, { 8, 8 });
        // std::cout << x.sizes() << std::endl;
        x = classifier->forward(x);
        return x;
    }

};
TORCH_MODULE(OtraNet);
