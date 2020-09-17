#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "Arenero.h"

//#include "_DenseNet.hpp"
//#include "_VGG.hpp"

int main(int argc, const char* argv[]) {

    //std::vector<int64_t> VEC = { 6, 12, 24, 16 };


    //_DenseBlock NN(5, 3, 32, 0.23);
    //_Transition NN1(5 * 32 + 3, 14);

    //_DenseNet   NN2(2, 32, VEC, 64, 0.213);

    //auto ui = torch::randn({ 2,3, 220,220 });
    //std::cout << NN << std::endl;
    //std::cout << NN2 << std::endl;

    //auto D = NN2->forward(ui);
    //std::cout << D.sizes() << std::endl;


    //auto T = NN->forward(ui);
    //std::cout << T.sizes() << std::endl;

    //auto D = NN1->forward(T);
    //std::cout << D.sizes() << std::endl;

    //std::cout << "";

    //auto x = torch::randn({ 12,3,10,10 });
    //std::cout << x.sizes() << std::endl;
    //auto  y = torch::adaptive_avg_pool2d(x, { 7,7});
    //std::cout << y.sizes() << std::endl;


    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

        uint64_t OUT_CLASSES = 2;
        if (CmdLineOpt::type_net == CmdLineOpt::TYPE::DENSENET) {
            _DenseNet NET(
                2,
                CmdLineOpt::growth_rate,
                CmdLineOpt::params,
                64,
                CmdLineOpt::drop_rate
            );

            Trainer<_DenseNet> TRAINER(NET);

        }
        else if (CmdLineOpt::type_net == CmdLineOpt::TYPE::OTRANET) {
            OtraNet NET(
                CmdLineOpt::params,
                CmdLineOpt::drop_rate,
                CmdLineOpt::batch_norm
            );

            Trainer<OtraNet> TRAINER(NET);
        }

        //else if (CmdLineOpt::type_net == CmdLineOpt::TYPE::PIRAMIDAL) {
        //    //VGG11 NET;
        //    //Trainer<VGG11> TRAINER(NET);
        //}
        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};
