#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"


int main(int argc, const char* argv[]) {
    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

        Trainer TRAINER;

        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};
