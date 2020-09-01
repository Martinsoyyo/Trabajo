#include "pch.h"

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
