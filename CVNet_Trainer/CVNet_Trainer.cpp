#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "DenseNet.h"


int main(int argc, const char* argv[]) {
    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);
        
        //auto ui = torch::randn({ 1,3, 64,64 });
        //Network3 net;
        //std::cout << net->forward(ui);

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
            OtraNet NET(CmdLineOpt::params, CmdLineOpt::drop_rate);
            Trainer<OtraNet> TRAINER(NET);
        }
        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};
