#pragma once
#include "cmdlineopt.h"
#include "CVNet.h"
#include "DenseNet.h"

#define DSEP "\\"
#define IMG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "IMAGES.tensor"
#define TRG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "TARGET.tensor"

class Trainer
{
public:
	Trainer();

	float Test(torch::Tensor& IMG, torch::Tensor& TRG);
	void Train(const uint32_t& EPOCH, torch::optim::Optimizer& OPT, torch::Tensor& IMG, torch::Tensor& TRG);

private:
	//Network NET;
	DenseNet NET;
	torch::Tensor _image, _target; 
};