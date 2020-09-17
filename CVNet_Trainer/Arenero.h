#pragma once
#include "pch.h"

struct _DenseLayerImpl : torch::nn::SequentialImpl {
    _DenseLayerImpl(
        int64_t CHANNEL_IN,
        int64_t GROWTH,
        float DROP_RATE)
    {
        using namespace torch::nn;

        push_back("norm1", BatchNorm2d(CHANNEL_IN));
        push_back("relu1", Functional(torch::relu));
        push_back("conv1", Conv2d(Conv2dOptions(CHANNEL_IN, 128, 1).stride(1).bias(false)));
        push_back("norm2", BatchNorm2d(128));
        push_back("relu2", Functional(torch::relu));
        push_back("conv2", Conv2d(Conv2dOptions(128, GROWTH, 3).stride(1).padding(1).bias(false)));
        if (DROP_RATE > 0) push_back("dropout", torch::nn::Dropout(DROP_RATE));
    };

    torch::Tensor forward(torch::Tensor x) {
        return torch::cat({ x, torch::nn::SequentialImpl::forward(x) }, 1);
    }
};
TORCH_MODULE(_DenseLayer);
    
struct _DenseBlockImpl : torch::nn::SequentialImpl {
    _DenseBlockImpl(
        int64_t LAYERS,
        int64_t CHANNEL_IN,
        int64_t GROWTH,
        float DROP_RATE)
    {
        for (int64_t i = 0; i < LAYERS; ++i) {
            push_back("denselayer" + std::to_string(i + 1), _DenseLayer(CHANNEL_IN + i * GROWTH, GROWTH, DROP_RATE));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_DenseBlock);

struct _TransitionImpl : torch::nn::SequentialImpl {
    _TransitionImpl(int64_t CHANNEL_IN)
    {
        using namespace torch::nn;
        push_back("norm", BatchNorm2d(CHANNEL_IN));
        push_back("relu ", Functional(torch::relu));
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
        const std::vector<size_t> LAYER_CONFIG,
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
        push_back(Linear(last, OUT_CLASSES));
        push_back(LogSoftmax(1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return SequentialImpl::forward(x);
    }
};
TORCH_MODULE(_DenseNet);

//
//struct _Dense2Impl : torch::nn::SequentialImpl {
//    _Dense2Impl(std::vector<uint32_t>& PARAMS, const float& DROP_RATE)
//    {
//        using namespace torch::nn;
//
//        auto CHANNEL_IN = PARAMS[0];
//        auto LAYERS = PARAMS[1];
//        auto GROWTH = PARAMS[2];
//
//        int64_t _in = CHANNEL_IN;
//        for (int i = 0; i < 6; i++) {
//
//            push_back(_Dense1(_in, LAYERS, GROWTH, DROP_RATE));
//            push_back(BatchNorm2d(_in + LAYERS * GROWTH));
//            push_back(Functional(torch::relu));
//            push_back(MaxPool2d(2));
//
//            _in += LAYERS * GROWTH;
//        }
//
//        push_back(Conv2d(Conv2dOptions(_in, 2, 3).stride(1).padding(1).bias(false)));
//        push_back(Flatten());
//        push_back(LogSoftmax(1));
//    }
//
//    torch::Tensor forward(torch::Tensor x) { return SequentialImpl::forward(x); }
//};
//TORCH_MODULE(_Dense2);


//
//struct _TransitionImpl : SequentialImpl {
//    _TransitionImpl(int64_t num_input_features, int64_t num_output_features) {
//        push_back("norm", BatchNorm2d(num_input_features));
//        push_back("relu ", Functional(torch::relu));
//        push_back("conv", Conv2d(Conv2dOptions(num_input_features, num_output_features, 1).stride(1).bias(false)));
//        push_back("pool", Functional(AvgPool2d(AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))));
//    }
//
//    // El foward() siempre tiene que estar definido aunque no haga nada y solo pase el metodo heredado, sino no ANDA!
//    torch::Tensor forward(torch::Tensor x) {
//        return torch::nn::SequentialImpl::forward(x);
//    }
//};
//TORCH_MODULE(_Transition);
//
//struct DenseNetImpl : torch::nn::Module {
//    torch::nn::Sequential features{ nullptr };
//    torch::nn::Linear classifier{ nullptr };
//
//    DenseNetImpl(
//        int64_t num_classes,
//        int64_t growth_rate,
//        std::vector<uint32_t> block_config,
//        int64_t num_init_features,
//        int64_t bn_size,
//        float drop_rate)
//    {
//        // First convolution
//        features = Sequential();
//        features->push_back("conv0", Conv2d(Conv2dOptions(3, num_init_features, 7).stride(2).padding(3).bias(false)));
//        features->push_back("norm0", BatchNorm2d(num_init_features));
//        features->push_back("relu0", Functional(torch::relu));
//        features->push_back("pool0", Functional(torch::max_pool2d, 3, 2, 1, 1, false));
//
//        // Para cada _DenseBlock
//        auto num_features = num_init_features;
//        for (size_t i = 0; i < block_config.size(); ++i) {
//            auto num_layers = block_config[i];
//            _DenseBlock block(num_layers, num_features, bn_size, growth_rate, drop_rate);
//            features->push_back("denseblock" + std::to_string(i + 1), block);
//            num_features = num_features + num_layers * growth_rate;
//
//            if (i != block_config.size() - 1) {
//                auto trans = _Transition(num_features, num_features / 2);
//                features->push_back("transition" + std::to_string(i + 1), trans);
//                num_features = num_features / 2;
//            }
//        }
//
//        // Final batch norm
//        features->push_back("norm5",BatchNorm2d(num_features));
//       
//        // Linear layer
//        classifier = Linear(num_features, num_classes);
//
//        register_module("features", features);
//        register_module("classifier", classifier);
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        auto features = this->features->forward(x);
//        auto out = torch::relu_(features);
//        out = torch::adaptive_avg_pool2d(out, { 1, 1 });
//
//        out = out.view({ features.size(0), -1 });
//        out = this->classifier->forward(out);
//
//        out = torch::log_softmax(out, 1);
//        return out;
//    };
//
//};
//TORCH_MODULE(DenseNet);
