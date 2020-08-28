#include "pch.h"
#include "cmdlineopt.h"

namespace CmdLineOpt {

    int  epoch                  = 10;
    int  batch_size             = 16;
    bool overwrite              = false;
    bool gpu                    = false;
    bool verbose                = false;
    float percent_to_train      = 0.75f;

    float drop_rate             = 0.345f;
    std::vector<uint32_t> densenet_params;
    uint32_t growth_rate        = 12;
    
    std::string dataset_path    = "";
    std::string dataset_prefix  = "";

    void CmdLineOpt(int argc, const char* argv[]) {
        try {
            cxxopts::Options options(argv[0], " - example command line options");

            options
                .positional_help("[optional args]")
                .show_positional_help()
                //.parse_positional({ "first", "second", "last" })
                ;
            // --input=E:\data\CementCrack\5y9wdsg2zt-1 --prefix=32x32(0.15) --training_percentage=0.15 --size_x=32 --size_y=32 --verbose 
            options
                .allow_unrecognised_options()
                .add_options()
                ("h,help", "Print help")
                ("GPU", "Habilita en uso de la GPU, si existe.", cxxopts::value<bool>(gpu))
                ("b,batch_size", "Numero de imagenes que entrena de manera simultanea. [BATCH SIZE]", cxxopts::value<int>(batch_size))
                ("p,path", "Direccion donde se encuentran las IMAGENES", cxxopts::value<std::string>(dataset_path))
                ("prefix", "Prefijo para los nombres de los archivos de dataset '*.tensor' (ej. 'PREFIJO_TRAIN_IMAGES.tensor'", cxxopts::value<std::string>(dataset_prefix))
                ("v,verbose", "Informacion detallada mientras se ejecuta.", cxxopts::value<bool>(verbose))
                ("e,epoch", "Numero de pasadas por el Lote de entrenamiento.", cxxopts::value<int>(epoch))
                ("t,train", "% del DATASET que se usa para entrenamiento.", cxxopts::value<float>(percent_to_train))
                ("o,overwrite", "carga el modelo de la RED si lo encuentra.", cxxopts::value<bool>(overwrite))

                ("m,modules", "Cantidad de capas internas en cada estadio de la Densenet.", cxxopts::value< std::vector<uint32_t>>(densenet_params))
                ("g,growth_rate", "Cantidad de capas que sa van agregando en cada etapa de la Densenet.", cxxopts::value< std::vector<uint32_t>>(densenet_params))
                ("d,drop_rate", "Drop_rate.", cxxopts::value<bool>(overwrite))

                ;

            auto result = options.parse(argc, argv);
            if (result.count("train")) std::cout << "[TRAINNING PERCENTAGE] " << percent_to_train << "%" << std::endl;
            if (result.count("epoch")) std::cout << "[EPOCH] " << epoch << std::endl;
            if (result.count("GPU")) std::cout << "[GPU] <ON>" << std::endl;
            if (result.count("batch_size")) std::cout << "[BATCH SIZE] " << batch_size << std::endl;
            if (result.count("verbose")) std::cout << "[VERBOSE] ON" << std::endl;
            if (result.count("path")) std::cout << "[IMAGE PATH] " << dataset_path << "" << std::endl;
            if (result.count("overwrite")) std::cout << "[LOAD MODEL] ON" << std::endl;
            if (result.count("growth_rate")) std::cout << "[GROWTH_RATE] " << growth_rate << std::endl;

            if (result.count("modules")) {
                std::cout << "Numero de Etapas en cada capa de la DenseNet <";
                for (const auto& IT : densenet_params) {
                    std::cout << IT << ",";
                }
                std::cout << ">" << std::endl;
            };

            if (result.count("prefix")) {
                dataset_prefix.append("_");
                std::cout << "Prefijo Dataset. <" << dataset_prefix << ">" << std::endl;
            }

            if (result.count("help")) {
                std::cout << options.help({ "", "Group" }) << std::endl;
                exit(0);
            }
            std::cout << std::endl;

        }
        catch (const cxxopts::OptionException& e) {
            std::cout << "CmdLineOpt error: " << e.what() << std::endl;
            exit(1);
        }
    };
}
