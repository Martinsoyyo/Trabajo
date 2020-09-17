#pragma once
#include "pch.h"

struct _DenseLayerImpl : torch::nn::SequentialImpl {
    _DenseLayerImpl(
        size_t CHANNEL_IN,
        size_t GROWTH,
        float DROP_RATE)
    {
        using namespace torch::nn;

        push_back("norm1", BatchNorm2d(CHANNEL_IN));
        push_back("relu1", Functional(torch::relu));
        push_back("conv1", Conv2d(Conv2dOptions(CHANNEL_IN, 128, 1).stride(1).bias(false)));
        push_back("norm2", BatchNorm2d(128));
        push_back("relu2", Functional(torch::relu));
        push_back("conv2", Conv2d(Conv2dOptions(128, GROWTH, 3).stride(1).padding(1).bias(false)));
        if (DROP_RATE > 0) push_back("dropout", Dropout(DROP_RATE));
    };

    torch::Tensor forward(torch::Tensor x) {
        return torch::cat({ x, torch::nn::SequentialImpl::forward(x) }, 1);
    }
};
TORCH_MODULE(_DenseLayer);
    
struct _DenseBlockImpl : torch::nn::SequentialImpl {
    _DenseBlockImpl(
        size_t LAYERS,
        size_t CHANNEL_IN,
        size_t GROWTH,
        float DROP_RATE)
    {
        for (auto i = 0; i < LAYERS; ++i) {
            push_back("denselayer" + std::to_string(i + 1), _DenseLayer(CHANNEL_IN + i * GROWTH, GROWTH, DROP_RATE));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_DenseBlock);

struct _TransitionImpl : torch::nn::SequentialImpl {
    _TransitionImpl(size_t CHANNEL_IN)
    {
        using namespace torch::nn;
        push_back("norm", BatchNorm2d(CHANNEL_IN));
        push_back("relu", Functional(torch::relu));
        push_back("conv", Conv2d(Conv2dOptions(CHANNEL_IN, CHANNEL_IN / 2, 1).stride(1).bias(false)));
        push_back("pool", Functional(AvgPool2d(AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))));
    }

    torch::Tensor forward(torch::Tensor x) {
        return SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_Transition);

struct _DenseNetImpl : torch::nn::SequentialImpl {
    _DenseNetImpl(
        size_t OUT_CLASSES,
        size_t GROWTH,
        const std::vector<int> LAYER_CONFIG,
        size_t IN_CLASSES,
        float DROP_RATE)
    {
        using namespace torch::nn;

        // Primera ETAPA transforma de (220 x 220 x 3) a (55 x 55x IN_CLASSES)
        push_back("conv0", Conv2d(Conv2dOptions(3, IN_CLASSES, 7).stride(2).padding(3).bias(false)));
        push_back("norm0", BatchNorm2d(IN_CLASSES));
        push_back("relu0", Functional(torch::relu));
        push_back("pool0", Functional(torch::max_pool2d, 3, 2, 1, 1, false));

        // Segunda ETAPA, parte principal de la Densenet
        // https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a
        size_t count = 0;
        size_t last = IN_CLASSES;
        for (auto IDX : LAYER_CONFIG) {
            push_back("_DenseBlock" + std::to_string(count), _DenseBlock(IDX, last, GROWTH, DROP_RATE));
            last = last + GROWTH * IDX;
            push_back("_Transition" + std::to_string(count++), _Transition(last));
            last = last / 2;
        }

        // Tercera ETAPA, parte de adapta a un perceptron de salida que termina en OUT_CLASSES
        push_back("final_norm", torch::nn::BatchNorm2d(last));
        push_back(MaxPool2d(3));
        push_back(Flatten());
        push_back(Linear(last, 256));
        push_back(Functional(torch::relu));
        push_back(Dropout(0.2));
        push_back(Linear(256, 2));
        push_back(LogSoftmax(1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_DenseNet);
