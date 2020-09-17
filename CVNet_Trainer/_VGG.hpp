#pragma once
#include "pch.h"

using namespace torch::nn;

static std::unordered_map<char, std::vector<int>> cfgs = {
  {'A', {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
  {'B', {64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
  {'D', {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
  {'E', {64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1, 512, 512, 512, 512, -1,  512, 512, 512, 512, -1}} };

Sequential makeLayers(
    const std::vector<int>& cfg,
    bool batch_norm = false) {
    torch::nn::Sequential seq;
    auto channels = 3;

    for (const auto& V : cfg) {
        if (V <= -1)
            seq->push_back(torch::nn::MaxPool2d(2));
        else {
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, V, 3).padding(1)));
            if (batch_norm) seq->push_back(torch::nn::BatchNorm2d(V));
            seq->push_back(torch::nn::Functional(torch::relu));

            channels = V;
        }
    }

    return seq;
}

struct VGGImpl : Module {
    Sequential features{ nullptr }, classifier{ nullptr };

    void _initialize_weights() {
        for (auto& module : modules(/*include_self=*/false)) {
            if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                init::kaiming_normal_(
                    M->weight,
                    /*a=*/0,
                    torch::kFanOut,
                    torch::kReLU);
                torch::nn::init::constant_(M->bias, 0);
            }
            else if (
                auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
                init::constant_(M->weight, 1);
                init::constant_(M->bias, 0);
            }
            else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                init::normal_(M->weight, 0, 0.01);
                init::constant_(M->bias, 0);
            }
        }
    };

    VGGImpl(Sequential features, int64_t num_classes = 2, bool initialize_weights = true) {
        classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            Functional(torch::relu),
            Dropout(),
            Linear(4096, 4096),
            Functional(torch::relu),
            Dropout(),
            Linear(4096, num_classes),
            LogSoftmax(1)
        );

        this->features = features;

        register_module("features", this->features);
        register_module("classifier", classifier);

        if (initialize_weights)  _initialize_weights();
    };

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = torch::adaptive_avg_pool2d(x, { 7, 7 });
        x = x.view({ x.size(0), -1 });
        x = classifier->forward(x);
        return x;
    };
};
TORCH_MODULE(VGG);

struct VGG11Impl : VGGImpl {
    VGG11Impl(int64_t num_classes = 2, bool initialize_weights = true)
        : VGGImpl(makeLayers(cfgs['A']), num_classes, initialize_weights) {};
};
TORCH_MODULE(VGG11);