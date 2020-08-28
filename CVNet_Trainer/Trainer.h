#pragma once
#include "cmdlineopt.h"
#include "CVNet.h"
#include "DenseNet.h"

class Trainer
{
public:
	Trainer();

	float Test(torch::Tensor& IMG, torch::Tensor& TRG);
	void Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG);

private:
	Network3 NET;
	//DenseNet NET;
	torch::Tensor _image, _target; 
};
