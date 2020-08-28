#pragma once
#include "cmdlineopt.h"
#include "CVNet.h"
#include "DenseNet.h"

class Trainer
{
public:
	Trainer();

private: // NOTA TEMPORARIA: No se usan desde fuera, no tienen necesidad de se pública
	float Test(torch::Tensor& IMG, torch::Tensor& TRG);
	void Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG);

private:
	//Network NET;
	DenseNet NET;
	torch::Tensor _image, _target; 
};
