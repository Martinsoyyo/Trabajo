#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "Arenero.h"

int main(int argc, const char* argv[]) {

    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

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
        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};
