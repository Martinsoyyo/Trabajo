#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "_DenseNet.h"
#include "VGG.hpp"

int main(int argc, const char* argv[]) {

    //std::vector<int> VEC = { 64 ,64, -1, 64, 64, -1, 64, 64, -1, 64, 64, -1, 64, 64, -1, 32, 32, -1 };
    //OtraNet NN(VEC, 0.21f, CmdLineOpt::batch_norm);

    //DenseNet121 NN(2);
    //auto ui = torch::randn({ 12,3, 220,220 });
    //std::cout << NN << std::endl;
    //std::cout << NN->forward(ui).sizes() << std::endl;
    //std::cout << NN->forward(ui).dtype() << std::endl;
    //std::cout << "";

    //auto x = torch::randn({ 12,3,10,10 });
    //std::cout << x.sizes() << std::endl;
    //auto  y = torch::adaptive_avg_pool2d(x, { 7,7});
    //std::cout << y.sizes() << std::endl;


    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

        if (CmdLineOpt::type_net == CmdLineOpt::TYPE::DENSENET) {
            if (CmdLineOpt::params[0] == 121) {
                DenseNet121 net(2);
                Trainer<DenseNet121> trainer(net);
            }
            else if (CmdLineOpt::params[0] == 169) {
                DenseNet169 net(2);
                Trainer<DenseNet169> trainer(net);
            }
            else if (CmdLineOpt::params[0] == 201) {
                DenseNet201 net(2);
                Trainer<DenseNet201> trainer(net);
            }
            else if (CmdLineOpt::params[0] == 161) {
                DenseNet161 net(2);
                Trainer<DenseNet161> trainer(net);
            }

        }
        else if (CmdLineOpt::type_net == CmdLineOpt::TYPE::OTRANET) {
            OtraNet NET(
                CmdLineOpt::params,
                CmdLineOpt::drop_rate,
                CmdLineOpt::batch_norm
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
