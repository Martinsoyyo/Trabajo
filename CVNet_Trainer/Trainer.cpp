#include "pch.h"
#include "Trainer.h"

using namespace std;
// es un espacio de nombres estandar muy usadado. Se abrevian comandos muy repetidos. No hay posibilidad real de confundirlo con otro entorno (espacio de nombres)

constexpr auto DSEP="/";
// Estandar de separador de carpetas compatible Win y Linux que puede convivir con la barra invertida de windows

#define IMG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "IMAGES.tensor"
#define TRG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "TARGET.tensor"
// ^-- Son inherentes a la implementación, el header debería casi exclusivamente a definición e interfaz con los usuarios/clientes

Trainer::Trainer(): 
	NET(
		2, //num_classes  
		CmdLineOpt::growth_rate,//growth_rate
		CmdLineOpt::densenet_params,//block_config
		64,//num_init_features
		4, //bn_size
		0 //drop_rate
	)
{
    try {
		cout << "Cargando " << IMG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix) << endl;
	    torch::load(_image, IMG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));
    	torch::load(_target, TRG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));
    } catch(exception &e) {
        cerr << "Trainer::Trainer() - torch::load" << endl << e.what();
        throw(e);
    }
    
	// Como todo el programa se basa en estos tensores, y solo uso "vistas" a ellos,
	torch::Device DeviceType = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
		// NOTA TEMPORARIA. Sacar de esta sección
    std::cout << "Device: ";
    if (DeviceType == torch::kCUDA) cout << "CUDA"; else cout << "CPU"; cout << endl;

	if (CmdLineOpt::gpu) {
	    cout <<  "Using CUDA GPU" << endl;
		_image  = _image.to(DeviceType);
		_target = _target.to(DeviceType);
		NET->to(DeviceType);
	} else cout <<  "Using CPU" << endl;

	// torch::optim::Adam optimizer(NET->parameters(), torch::optim::AdamOptions(1e-5).weight_decay(0.005).beta1(0.9).beta2(0.999));
    // Según la versión de Pytorch utilizada. Habría que ver cómo hacerlo en tiempo de compilación
	//torch::optim::Adam optimizer(NET->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
	torch::optim::Adam optimizer(NET->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.9, 0.995)));	
	
	std::cout << NET << std::endl;

	// Referencias que apuntan a la zona de testeo y entrenamiento.
	const auto N = int64_t(CmdLineOpt::percent_to_train * _image.size(0));

	auto image_train = _image.slice(0, 0, N);
	auto image_test = _image.slice(0, N + 1);

	auto target_train = _target.slice(0, 0, N);
	auto target_test = _target.slice(0, N + 1);

	// Carga la RED si existe.
	if (CmdLineOpt::overwrite) {  // NOTA TEMPORARIA. La idea era que el overwrite IGNORE el archivos. Revisar y poner en común la semántica de las opciones.
		try {
			torch::load(NET, "model.pt"); // NOTA TEMPORARIA. Considerar el control de que el modelo cargado coincida con el del código.
			std::cout << "MODEL Loaded..." << std::endl;
		}
		catch (...) {
			std::cout << "MODEL Not Found... OVERWRITE ignored!." << std::endl;
		}
	} else{
		std::cout << "MODEL Created..." << std::endl;
	}

	// Mido el tiempo que tarda en analizar una imagen en promedio.
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

		auto best_result = Test(image_test, target_test);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	std::cout << "Tiempo de analisis por cada imagen: " << elapsed_seconds.count()/ image_test.size(0) << "s\n";

	// Empieza el entrenamiento.
	for (int i = 0; i < CmdLineOpt::epoch; i++)
	{
		Train(i, optimizer, image_train, target_train);
		Test(image_train, target_train); //Testeo sobre lo que va entrenado.

		auto result = Test(image_test, target_test); //Testeo LOTE que no aprende la RED.
		if (result > best_result) {
			torch::save(NET, "model.pt");
			best_result = result;
		}
	}

	std::cout << "Best Test Batch: " << best_result << std::endl;
	
	// Recupero la red que se entrena con GPU y la guardo en un formato que entienda la CPU, sino no la puedo recuperar en Windows.
	if (DeviceType == torch::kCUDA) {
		std::cout << "Adaptando de GPU a CPU" << std::endl;
		torch::load(NET, "model.pt");
		NET->to(torch::kCPU);
		torch::save(NET, "model_cpu.pt");
	};
}

void Trainer::Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG) {
	std::cout << "Trainning... ";
	NET->train();

	// Separo los tensores de entrada en Batches.
	auto IMAGE  = IMG.split(CmdLineOpt::batch_size);
	auto TARGET = TRG.split(CmdLineOpt::batch_size);

	for (uint32_t idx = 0; idx < IMAGE.size(); idx++){
		OPT.zero_grad();
			auto prediction = NET->forward(IMAGE[idx].to(at::kFloat).div_(255));
			//auto loss = torch::binary_cross_entropy(prediction, TARGET[idx].to(at::kLong));
			auto loss = torch::nll_loss(prediction, TARGET[idx].to(at::kLong));
			loss.backward();
		OPT.step();

		std::printf("\rTrain Epoch: %u [%5u/%5lu] Loss: %.4f ",
			EPOCH, idx * CmdLineOpt::batch_size,
			IMG.size(0),
			loss.template item<float>());
	}
	std::cout << std::endl;
}

float Trainer::Test(torch::Tensor& IMG, torch::Tensor& TRG) {
	std::cout << "Testing... ";
	NET->eval();

	// Separo los tensores de entrada en Batches.
	auto IMAGE = IMG.split(CmdLineOpt::batch_size);
	auto TARGET = TRG.split(CmdLineOpt::batch_size);

	size_t correct = 0;
	for (uint32_t idx = 0; idx < IMAGE.size(); idx++) {
		auto prediction = NET->forward(IMAGE[idx].to(at::kFloat).div_(255));
		correct += prediction.argmax(1).eq(TARGET[idx]).sum().template item<int64_t>();
	};

	float ACCURACY = (float)correct / (float)IMG.size(0);
	std::cout << "Accuracy " << ACCURACY << std::endl;
	return ACCURACY;
};

//void Trainer::Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG) {
//	std::cout << "Trainning... ";
//	NET->train();
//
//	// Separo los tensores de entrada en Batches.
//	auto IMAGE = IMG.split(CmdLineOpt::batch_size);
//	auto TARGET = TRG.split(CmdLineOpt::batch_size);
//
//	for (uint32_t idx = 0; idx < IMAGE.size(); idx++) {
//		OPT.zero_grad();
//		auto prediction = NET->forward(IMAGE[idx].to(at::kFloat).div_(255));
//		torch::Tensor loss = torch::binary_cross_entropy(prediction, TARGET[idx].to(at::kFloat));
//		loss.backward();
//		OPT.step();
//
//		std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f ",
//			EPOCH, idx * CmdLineOpt::batch_size,
//			IMG.size(0),
//			loss.template item<float>());
//	}
//	std::cout << std::endl;
//}

//void Trainer::Test(torch::Tensor& IMG, torch::Tensor& TRG) {
//	std::cout << "Testing... ";
//	NET->eval();
//
//	// Separo los tensores de entrada en Batches.
//	auto IMAGE = IMG.split(CmdLineOpt::batch_size);
//	auto TARGET = TRG.split(CmdLineOpt::batch_size);
//
//	size_t correct = 0;
//	for (uint32_t idx = 0; idx < IMAGE.size(); idx++) {
//		auto prediction = NET->forward(IMAGE[idx].to(at::kFloat).div_(255));
//		correct += TARGET[idx].to(at::kFloat).eq(prediction.squeeze_().round_()).sum().template item<int64_t>();
//	};
//
//	float ACCURACY = (float)correct / (float)IMG.size(0);
//	std::cout << "Accuracy " << ACCURACY << std::endl;
//};
