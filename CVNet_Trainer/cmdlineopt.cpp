#include "pch.h"
#include "cmdlineopt.h"

namespace CmdLineOpt {
    size_t  epoch               = 10;
    size_t  batch_size          = 32;
    size_t overwrite            = 1;
    size_t batch_norm           = 1;
    bool cpu                    = false;
    bool verbose                = false;
    float percent_to_train      = 0.75f;
    float drop_rate             = 0.145f;
    float learning_rate         = 0.0001f;

    size_t type_net;
    std::string net_name;
    std::vector<int> params;
    uint64_t growth_rate        = 12;
    
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
            options
                .allow_unrecognised_options()
                .add_options()
                ("h,help", "Print help")
                ("CPU", "Force CPU.", cxxopts::value<bool>(cpu))
                ("b,batch_size", "Numero de imagenes que entrena de manera simultanea. [BATCH SIZE]", cxxopts::value<size_t>(batch_size))
                ("p,path", "Direccion donde se encuentran las IMAGENES", cxxopts::value<std::string>(dataset_path))
                ("prefix", "Prefijo para los nombres de los archivos de dataset '*.tensor' (ej. 'PREFIJO_TRAIN_IMAGES.tensor'", cxxopts::value<std::string>(dataset_prefix))
                ("v,verbose", "Informacion detallada mientras se ejecuta.", cxxopts::value<bool>(verbose))
                ("e,epoch", "Numero de pasadas por el Lote de entrenamiento.", cxxopts::value<size_t>(epoch))
                ("t,train", "% del DATASET que se usa para entrenamiento.", cxxopts::value<float>(percent_to_train))
                ("o,overwrite", "carga el modelo de la RED si lo encuentra.", cxxopts::value<size_t>(overwrite))
                ("l,learning_rate", "Learning Rate.", cxxopts::value<float>(learning_rate))
                ("g,growth_rate", "Cantidad de capas que sa van agregando en cada etapa. <DENSENET>", cxxopts::value<size_t>(growth_rate))
                ("params", "Parametros de la RED.", cxxopts::value< std::vector<int>>(params))
                ("d,drop_rate", "DropRate %", cxxopts::value<float>(drop_rate))
                ("use", "Que tipo de RED uso.", cxxopts::value<std::string>(net_name))
                ("batch_norm", "batch_norm.", cxxopts::value<size_t>(batch_norm))

                ;

            auto result = options.parse(argc, argv);
            if (result.count("train")) std::cout << "[TRAINNING PERCENTAGE] " << percent_to_train << "%" << std::endl;
            if (result.count("epoch")) std::cout << "[EPOCH] " << epoch << std::endl;
            if (result.count("CPU")) std::cout << "[forced CPU] <ON>" << std::endl;
            if (result.count("batch_size")) std::cout << "[BATCH SIZE] " << batch_size << std::endl;
            if (result.count("verbose")) std::cout << "[VERBOSE] ON" << std::endl;
            if (result.count("path")) std::cout << "[IMAGE PATH] " << dataset_path << std::endl;
            if (result.count("overwrite")) std::cout << "[LOAD MODEL] " << overwrite << std::endl;
            if (result.count("growth_rate")) std::cout << "[GROWTH_RATE] " << growth_rate << std::endl;
            if (result.count("drop_rate")) std::cout << "[DROP_RATE] " << drop_rate << "%" << std::endl;
            if (result.count("learning_rate")) std::cout << "[LEARNING_RATE] " << learning_rate << std::endl;
            if (result.count("batch_norm")) std::cout << "[BATCH_NORM] =" << batch_norm << std::endl;

            if (result["use"].as<std::string>() == "densenet") {
                type_net = TYPE::DENSENET;
                std::cout << "[DENSENET] ";
                std::cout << " <";
                for (const auto& IT : params) {
                    std::cout << IT << " ";
                }
                std::cout << ">" << std::endl;
            };

            if (result["use"].as<std::string>() == "otranet") {
                type_net = TYPE::OTRANET;
                std::cout << "[OTRANET] ";
                std::cout << " <";
                for (const auto& IT : params) {
                    std::cout << IT << " ";
                }
                std::cout << ">" << std::endl;
            };

            if (result["use"].as<std::string>() == "piramidal") {
                type_net = TYPE::PIRAMIDAL;
                std::cout << "[PIRAMIDAL] ";
                std::cout << " <";
                for (const auto& IT : params) {
                    std::cout << IT << " ";
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
