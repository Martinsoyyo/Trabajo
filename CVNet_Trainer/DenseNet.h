#pragma once
#include "pch.h"

using namespace torch::nn;

struct _DenseLayerImpl : SequentialImpl {
    _DenseLayerImpl(
        int64_t num_input_features,
        int64_t growth_rate,
        int64_t bn_size,
        float drop_rate)
    {
        push_back("norm1", BatchNorm2d(num_input_features));
        push_back("relu1", Functional(torch::relu));
        push_back("conv1", Conv2d(Conv2dOptions(num_input_features, bn_size * growth_rate, 1).stride(1).bias(false)));

        push_back("norm2", BatchNorm2d(bn_size * growth_rate));
        push_back("relu2", Functional(torch::relu));
        push_back("conv2", Conv2d(Conv2dOptions(bn_size * growth_rate, growth_rate, 3).stride(1).padding(1).bias(false)));

        if (drop_rate > 0.0f) push_back(Dropout(drop_rate));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::cat({ x, SequentialImpl::forward(x) }, 1);
    }
};
TORCH_MODULE(_DenseLayer);

struct _DenseBlockImpl : SequentialImpl {
    _DenseBlockImpl(
        int64_t num_layers,
        int64_t num_input_features,
        int64_t bn_size,
        int64_t growth_rate,
        float drop_rate)
    {
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate);
            push_back("denselayer" + std::to_string(i + 1), layer);
        }
    }

    // El foward() siempre tiene que estar definido aunque no haga nada y solo pase el metodo heredado, sino no ANDA!
    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_DenseBlock);

struct _TransitionImpl : torch::nn::SequentialImpl {
    _TransitionImpl(int64_t num_input_features, int64_t num_output_features) {
        push_back("norm", BatchNorm2d(num_input_features));
        push_back("relu ", Functional(torch::relu));
        push_back("conv", Conv2d(Conv2dOptions(num_input_features, num_output_features, 1).stride(1).bias(false)));
        push_back("pool", Functional(AvgPool2d(AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))));
    }

    // El foward() siempre tiene que estar definido aunque no haga nada y solo pase el metodo heredado, sino no ANDA!
    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_Transition);

struct DenseNetImpl : torch::nn::Module {
    torch::nn::Sequential features{ nullptr };
    torch::nn::Linear classifier{ nullptr };

    DenseNetImpl(
        int64_t num_classes,
        int64_t growth_rate,
        std::vector<int64_t> block_config,
        int64_t num_init_features,
        int64_t bn_size,
        float drop_rate)
    {
        // First convolution
        features = Sequential();
        features->push_back("conv0", Conv2d(Conv2dOptions(3, num_init_features, 7).stride(2).padding(3).bias(false)));
        features->push_back("norm0", BatchNorm2d(num_init_features));
        features->push_back("relu0", Functional(torch::relu));
        features->push_back("pool0", Functional(torch::max_pool2d, 3, 2, 1, 1, false));

        // Para cada _DenseBlock
        auto num_features = num_init_features;
        for (size_t i = 0; i < block_config.size(); ++i) {
            auto num_layers = block_config[i];
            _DenseBlock block(num_layers, num_features, bn_size, growth_rate, drop_rate);
            features->push_back("denseblock" + std::to_string(i + 1), block);
            num_features = num_features + num_layers * growth_rate;

            if (i != block_config.size() - 1) {
                auto trans = _Transition(num_features, num_features / 2);
                features->push_back("transition" + std::to_string(i + 1), trans);
                num_features = num_features / 2;
            }
        }

        // Final batch norm
        features->push_back("norm5",BatchNorm2d(num_features));
       
        // Linear layer
        classifier = Linear(num_features, num_classes);

        register_module("features", features);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto features = this->features->forward(x);
        auto out = torch::relu_(features);
        out = torch::adaptive_avg_pool2d(out, { 1, 1 });

        out = out.view({ features.size(0), -1 });
        out = this->classifier->forward(out);

        out = torch::log_softmax(out, 1);
        return out;
    };

};
TORCH_MODULE(DenseNet);
