#include "pch.h"
#include "cmdlineopt.h"
#include "Trainer.h"
#include "CVNet.h"
#include "DenseNet.h"

using namespace std;


int main(int argc, const char* argv[]) {
    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        // --path=/home/seba/Proyectos/data/CementCrack/5y9wddg2zt-1-sz20 --verbose --prefix=64 --size=64
        CmdLineOpt::CmdLineOpt(argc, argv);

        Trainer TRAINER;

        return 0;
    }
    catch (exception e) {
        cerr << "Error: " << e.what();
    }
};
