#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "DenseNet.h"
#include "VGG.hpp"


struct _Dense0Impl : torch::nn::SequentialImpl {
    _Dense0Impl(
        int64_t CHANNEL_IN,
        int64_t CHANNEL_OUT,
        float DROP_RATE = 0)
    {
        using namespace torch::nn;

        push_back(Conv2d(Conv2dOptions(CHANNEL_IN, CHANNEL_OUT, 3).stride(1).padding(1).bias(false)));
        push_back(BatchNorm2d(CHANNEL_OUT));
        push_back(Functional(torch::relu));
        if (DROP_RATE > 0.0f) push_back(Dropout(DROP_RATE));
    }

    torch::Tensor forward(torch::Tensor x) { return torch::cat({ x, SequentialImpl::forward(x) }, 1); }
};
TORCH_MODULE(_Dense0);

struct _Dense1Impl : torch::nn::SequentialImpl {
    _Dense1Impl(
        int64_t CHANNEL_IN,
        int64_t LAYERS,
        int64_t GROWTH,
        float DROP_RATE = 0)
    {
        using namespace torch::nn;

        int64_t _in = CHANNEL_IN;
        for (int64_t i = 0; i < LAYERS; i++) {
            push_back(_Dense0(_in, GROWTH, DROP_RATE));
            _in += GROWTH;
        }
    }

    torch::Tensor forward(torch::Tensor x) { return SequentialImpl::forward(x); }
};
TORCH_MODULE(_Dense1);

struct _Dense2Impl : torch::nn::SequentialImpl {
    _Dense2Impl(std::vector<uint32_t>& PARAMS, const float& DROP_RATE)
    {
        using namespace torch::nn;

        auto CHANNEL_IN = PARAMS[0];
        auto LAYERS = PARAMS[1];
        auto GROWTH = PARAMS[2];

        int64_t _in = CHANNEL_IN;
        for (int i = 0; i < 6; i++) {

            push_back(_Dense1(_in, LAYERS, GROWTH, DROP_RATE));
            push_back(BatchNorm2d(_in + LAYERS * GROWTH));
            push_back(Functional(torch::relu));
            push_back(MaxPool2d(2));

            _in += LAYERS * GROWTH;
        }

        push_back(Conv2d(Conv2dOptions(_in, 2, 3).stride(1).padding(1).bias(false)));
        push_back(Flatten());
        push_back(LogSoftmax(1));
    }

    torch::Tensor forward(torch::Tensor x) { return SequentialImpl::forward(x); }
};
TORCH_MODULE(_Dense2);



int main(int argc, const char* argv[]) {

    //std::vector<uint32_t> VEC = { 22,33};
    //OtraNet NN(VEC, 0.21f);

    //auto ui = torch::randn({ 1,3, 64,64 });
    //std::cout << NN << std::endl;
    //std::cout << NN->forward(ui).sizes() << std::endl;
    //std::cout << NN->forward(ui).dtype() << std::endl;

    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

        if (CmdLineOpt::type_net == CmdLineOpt::TYPE::DENSENET) {
            DenseNet NET(
                	2, //num_classes  
                	CmdLineOpt::growth_rate,//growth_rate
                	CmdLineOpt::params,//block_config
                	64,//num_init_features
                	4, //bn_size
                	0 //drop_rate
                );  

            Trainer<DenseNet> TRAINER(NET);
        }
        else if (CmdLineOpt::type_net == CmdLineOpt::TYPE::OTRANET) {
            OtraNet NET(
                CmdLineOpt::params,
                CmdLineOpt::drop_rate
            );

            Trainer<OtraNet> TRAINER(NET);
        }
        else if (CmdLineOpt::type_net == CmdLineOpt::TYPE::PIRAMIDAL) {
            //_Dense2 NET(
            //    CmdLineOpt::params,
            //    CmdLineOpt::drop_rate
            //);

            //VGG11 NET(2);
            //Trainer<VGG11> TRAINER(NET);
        }
        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};
