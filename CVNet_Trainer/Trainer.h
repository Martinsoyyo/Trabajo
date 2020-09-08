#pragma once
#include "pch.h"
#include "cmdlineopt.h"
#include "CVNet.h"
#include "DenseNet.h"

using namespace std;
// es un espacio de nombres estandar muy usadado. Se abrevian comandos muy repetidos. No hay posibilidad real de confundirlo con otro entorno (espacio de nombres)

constexpr auto DSEP = "/";
// Estandar de separador de carpetas compatible Win y Linux que puede convivir con la barra invertida de windows

#define IMG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "IMAGES.tensor"
#define TRG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "TARGET.tensor"
// ^-- Son inherentes a la implementación, el header debería casi exclusivamente a definición e interfaz con los usuarios/clientes


template<class NET>
class Trainer {
public:
	Trainer(const NET& OBJ);
	float Test(torch::Tensor& IMG, torch::Tensor& TRG);
	void Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG);

	void foo();
private:
	torch::Tensor _image, _target;
	NET _net;
};

template<class NET>
Trainer<NET>::Trainer(const NET& OBJ) :_net(OBJ) {
	foo();
};

template<class NET>
void Trainer<NET>::foo() {

	try {
		cout << "Cargando " << IMG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix) << endl;
		torch::load(_image, IMG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));
		torch::load(_target, TRG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));
	}
	catch (exception& e) {
		cerr << "Trainer::Trainer() - torch::load" << endl << e.what();
		throw(e);
	}

	torch::Device DeviceType = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	// NOTA TEMPORARIA. Sacar de esta sección
	std::cout << "Device: ";
	if (DeviceType == torch::kCUDA) cout << "CUDA disponible"; else cout << "CPU solamente"; cout << endl;

	if (!CmdLineOpt::cpu && DeviceType == torch::kCUDA) {
		cout << "Using CUDA GPU" << endl;
		_image = _image.to(DeviceType);
		_target = _target.to(DeviceType);
		_net->to(DeviceType);
	}
	else cout << "Using CPU" << endl;



	// Carga la RED si existe.
	if (CmdLineOpt::overwrite == true) {
		try {
			torch::load(_net, "model.pt"); // NOTA TEMPORARIA. Considerar el control de que el modelo cargado coincida con el del código.
			std::cout << "MODEL Loaded..." << std::endl;
		}
		catch (...) {
			std::cout << "MODEL Not Found... OVERWRITE ignored!." << std::endl;
		}
	}
	else {
		std::cout << "MODEL Created..." << std::endl;
	}

	torch::optim::Adam optimizer(
		_net->parameters(),
		torch::optim::AdamOptions(CmdLineOpt::learning_rate)
		.betas(std::make_tuple(0.9, 0.995))
		.eps(1e-8)
		.weight_decay(0.05)
	);

	std::cout << _net << std::endl;

	// Referencias que apuntan a la zona de testeo y entrenamiento.
	const auto N = int64_t(CmdLineOpt::percent_to_train * _image.size(0));

	auto image_train = _image.slice(0, 0, N);
	auto image_test = _image.slice(0, N + 1);

	auto target_train = _target.slice(0, 0, N);
	auto target_test = _target.slice(0, N + 1);

	// Mido el tiempo que tarda en analizar una imagen en promedio.
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	auto best_result = Test(image_test, target_test);
	//auto best_result = 0;
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "Tiempo de analisis por cada imagen: " << elapsed_seconds.count() / image_test.size(0) << "s\n";

	float rate = 0.05f;
	// Empieza el entrenamiento.
	for (int i = 0; i < CmdLineOpt::epoch; i++)
	{

		Train(i, optimizer, image_train, target_train);
		Test(image_train, target_train);//Testeo sobre lo que va entrenado.
		auto result = Test(image_test, target_test); //Testeo LOTE que no aprende la RED.
		std::cout << std::endl;

		//for (auto& group : optimizer.param_groups())
		//{
		//	if (group.has_options())
		//	{
		//		auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
		//		cout << "[ Learning_rate] " << options.lr() << endl;;
		//		options.lr(options.lr() * (1.0 - rate));
		//	}
		//}

	if (result > best_result) {
			torch::save(_net, "model.pt");
			best_result = result;
		}
	}

	std::cout << "Best Test Batch: " << best_result << std::endl;

	// Recupero la red que se entrena con GPU y la guardo en un formato que entienda la CPU, sino no la puedo recuperar en Windows.
	if (DeviceType == torch::kCUDA) {
		std::cout << "Adaptando de GPU a CPU" << std::endl;
		torch::load(_net, "model.pt");
		_net->to(torch::kCPU);
		torch::save(_net, "model_cpu.pt");
	};
}

template<class NET>
void Trainer<NET>::Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG) {
	std::cout << "Trainning... ";
	_net->train();

	// Separo los tensores de entrada en Batches.
	auto IMAGE = IMG.split(CmdLineOpt::batch_size);
	auto TARGET = TRG.split(CmdLineOpt::batch_size);

	for (uint32_t idx = 0; idx < IMAGE.size(); idx++) {
		OPT.zero_grad();
		auto prediction = _net->forward(IMAGE[idx].to(at::kFloat).div_(255));
		//auto loss = torch::binary_cross_entropy(prediction, TARGET[idx].to(at::kLong));
		auto loss = torch::nll_loss(prediction, TARGET[idx].to(at::kLong));
		loss.backward();
		OPT.step();

		std::printf("\rTrain Epoch: %u [%5u/%5lu] Loss: %.4f ",
			EPOCH, idx * CmdLineOpt::batch_size,
			IMG.size(0),
			loss.template item<float>());
	}
}

template<class NET>
float Trainer<NET>::Test(torch::Tensor& IMG, torch::Tensor& TRG) {
	std::cout << "Testing... ";
	_net->eval();

	auto IMAGE = IMG.split(CmdLineOpt::batch_size);
	auto TARGET = TRG.split(CmdLineOpt::batch_size);

	size_t correct = 0;
	for (uint32_t idx = 0; idx < IMAGE.size(); idx++) {
		auto prediction = _net->forward(IMAGE[idx].to(at::kFloat).div_(255));
		correct += prediction.argmax(1).eq(TARGET[idx]).sum().template item<int64_t>();
	};

	float ACCURACY = (float)correct / (float)IMG.size(0);
	std::cout << "Accuracy " << ACCURACY << " ";
	return ACCURACY;
};