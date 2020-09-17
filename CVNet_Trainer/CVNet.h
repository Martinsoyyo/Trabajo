#pragma once
#include "pch.h"

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


//--path C:\Repositories\CementCrack\prueba --prefix 64x64 --epoch 10 0 -b 12 -o 0 
// --use otranet --params 32,-1,32,-1,32,-1,32,-1 --drop_rate 0.22 --learning_rate 0.0001 --batch_norm 1

const size_t CHANNEL_IN = 3;
const size_t CHANNEL_OUT= 2;

struct OtraNetImpl : torch::nn::SequentialImpl
{
    torch::nn::Sequential features, classifier;

    OtraNetImpl(const std::vector<size_t>& PARAMS, const float& DROP_RATE, const bool& BATCH_NORM) {
        using namespace torch::nn;

        size_t count_reduction = 0;
        size_t channel = CHANNEL_IN;
        for (const auto& V : PARAMS)
        {
            if (V <= -1) {
                features->push_back(MaxPool2d(2));
                count_reduction++;
            }
            else {
                features->push_back(Conv2d(Conv2dOptions(channel, V, 3 /*kernel*/).padding(1).bias(true)));
                if (BATCH_NORM) features->push_back(BatchNorm2d(V));
                if (DROP_RATE) features->push_back(Dropout(DROP_RATE));
                features->push_back(Functional(torch::relu));
                channel = V;
            }
        };

        classifier = torch::nn::Sequential(
            Linear(channel * 8 * 8, 256),
            Functional(torch::relu),
            Dropout(0.5),
            Linear(256, 256),
            Functional(torch::relu),
            Dropout(0.5),
            Linear(256, CHANNEL_OUT),
            LogSoftmax(1)
        );

        register_module("features", this->features);
        register_module("classifier", classifier);

        _initialize_weights();
    }

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);

        x = torch::adaptive_avg_pool2d(x, { 8, 8 });
        x = x.view({ x.size(0), -1 });
        x = classifier->forward(x);
        return x;
    }

    void _initialize_weights() {
        for (auto& module : modules(/*include_self=*/false)) {
            if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                torch::nn::init::kaiming_normal_(
                    M->weight,
                    /*a=*/0,
                    torch::kFanOut,
                    torch::kReLU);
                torch::nn::init::constant_(M->bias, 0);
            }
            else if (
                auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
                torch::nn::init::constant_(M->weight, 1);
                torch::nn::init::constant_(M->bias, 0);
            }
            else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                torch::nn::init::normal_(M->weight, 0, 0.01);
                torch::nn::init::constant_(M->bias, 0);
            }
        }
    }

};
TORCH_MODULE(OtraNet);
