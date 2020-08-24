#include "pch.h"
#include "Trainer.h"

Trainer::Trainer(): 
	NET(
		2, //num_classes  
		16,//growth_rate
		std::vector<int64_t> { 3, 3, 3, 3 },//block_config
		64,//num_init_features
		4, //bn_size
		0.345f //drop_rate
	)
{
	torch::load(_image, IMG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));
	torch::load(_target, TRG_FNAME(CmdLineOpt::dataset_path, CmdLineOpt::dataset_prefix));

	// Como todo el programa se basa en estos tensores, y solo uso "vistas" a ellos,
	// esto deberia ser suficiente para poner todo el programa en GPU, VERIFICAR!
	if (CmdLineOpt::gpu) {
		torch::Device DeviceType = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
		_image  = _image.to(DeviceType);
		_target = _target.to(DeviceType);
		NET->to(DeviceType);
	};

	torch::optim::Adam optimizer(NET->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

	// Referencias que apuntan a la zona de testeo y entrenamiento.
	const auto N = int64_t(CmdLineOpt::percent_to_train * _image.size(0));

	auto image_train = _image.slice(0, 0, N);
	auto image_test = _image.slice(0, N + 1);

	auto target_train = _target.slice(0, 0, N);
	auto target_test = _target.slice(0, N + 1);


	for (int i = 0; i < CmdLineOpt::epoch; i++)
	{
		Train(i, optimizer, image_train, target_train);
		Test(image_train, target_train); //Testeo sobre lo que va entrenado.
		Test(image_test, target_test);	 //Testeo LOTE que no aprende la RED.
	}

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
			auto loss = torch::nll_loss(prediction, TARGET[idx].to(at::kLong));
			loss.backward();
		OPT.step();

		std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f ",
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