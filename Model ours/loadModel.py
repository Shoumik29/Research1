import torch
import torchvision
from torch import nn
from torchsummary import summary
from torchvision import models
import tensorflow as tf
from tensorflow.python.platform import gfile


def loading_model_weights():

	weights = []
	biases = []
	means = []
	variances = []

	torch_model = torch.load('wav2lip.pth')
	
	j=1
	for i in torch_model['state_dict']:
		if j==7:
			print(i)
			print(torch_model['state_dict'][i])
			break
		j+=1
			
loading_model_weights()
