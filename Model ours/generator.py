from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Activation, Concatenate, Conv2DTranspose, Add, Input, ReLU
from keras.models import Model
import torch
import numpy as np
from dataProcess import getitem
import matplotlib.pyplot as plt
import tensorflow as tf

import keras.backend as K







def loading_model_weights():

	weights = []
	biases = []
	gamma = []
	beta = []
	means = []
	variances = []

	torch_model = torch.load('wav2lip.pth')
	
	j = 0
	for i in torch_model['state_dict']:
		if j % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w = np.moveaxis(w, [0,1], [-1,-2])
			weights.append(w)
			#print(i," ",w.shape, '\n')
		
		if (j-1) % 7 == 0:
			biases.append(torch_model['state_dict'][i].cpu().numpy())
			#print(i," ",torch_model['state_dict'][i].shape, '\n')
		
		if (j-2) % 7 == 0:
			gamma.append(torch_model['state_dict'][i].cpu().numpy())
			
		if (j-3) % 7 == 0:
			beta.append(torch_model['state_dict'][i].cpu().numpy())
		
		if (j-4) % 7 == 0:
			means.append(torch_model['state_dict'][i].cpu().numpy())
			#print(i," ",torch_model['state_dict'][i].shape, '\n')
		
		if (j-5) % 7 == 0:
			variances.append(torch_model['state_dict'][i].cpu().numpy())
			#print(i," ",torch_model['state_dict'][i].shape, '\n')
		j += 1

	

	decoderModel = Decoder()	
	
	face_index = [1,2,4,5,7,8,11,12,15,17,21,23,29,31,37,39,45,47,51,53,59,61,67,69,73,75,81,83,89,91,95,97,103,105,109,111]
	
	audio_index = [18,20,24,26,32,34,40,42,46,48,54,56,62,64,68,70,76,78,84,86,90,92,98,100,104,106]
	
	decoder_index = [110,112,116,117,119,120,124,125,127,128,131,132,136,137,139,140,143,144,148,149,151,152,155,156,160,161,163,164,167,168,172,
	173,175,176,179,180,184,185,187]
	
	print(len(face_index)+len(audio_index)+len(decoder_index))
	
	
	
	cj=0
	bj=0
	cnt=0
	for i in face_index:
		
		try:
			if 'Conv2D' in str(decoderModel.layers[i]):
				decoderModel.layers[i].set_weights([weights[cj], biases[cj]])
				cj += 1
				cnt += 1
			if 'BatchNormalization' in str(decoderModel.layers[i]):
				decoderModel.layers[i].set_weights([gamma[bj], beta[bj], means[bj], variances[bj]])
				bj += 1
				cnt += 1
				
		except:
			print("pass")
			
	for j in audio_index:
		
		try:
			if 'Conv2D' in str(decoderModel.layers[j]):
				decoderModel.layers[j].set_weights([weights[cj], biases[cj]])
				cj += 1
				cnt += 1
			if 'BatchNormalization' in str(decoderModel.layers[j]):
				decoderModel.layers[j].set_weights([gamma[bj], beta[bj], means[bj], variances[bj]])
				bj += 1
				cnt += 1
		except:
			print("pass")	
			
	for k in decoder_index:
	
		try:
			if 'Conv2D' in str(decoderModel.layers[k]):
				decoderModel.layers[k].set_weights([weights[cj], biases[cj]])
				cj += 1
				cnt += 1
			if 'BatchNormalization' in str(decoderModel.layers[k]):
				decoderModel.layers[k].set_weights([gamma[bj], beta[bj], means[bj], variances[bj]])
				bj += 1
				cnt += 1
				
		except:
			print("pass")
	
	
	'''faceModel = faceEncoder()
	audioModel = audioEncoder()
	
	cj=0
	bj=0
	for i in range(1, len(faceModel.layers)):
		if 'Conv2D' in str(faceModel.layers[i]):
			faceModel.layers[i].set_weights([weights[cj], biases[cj]])
			cj += 1
		if 'BatchNormalization' in str(faceModel.layers[i]):
			faceModel.layers[i].set_weights([np.ones(means[bj].shape), np.zeros(means[bj].shape), means[bj], variances[bj]])
			bj += 1
			
	
	for i in range(1, len(audioModel.layers)):
		if 'Conv2D' in str(audioModel.layers[i]):
			audioModel.layers[i].set_weights([weights[cj], biases[cj]])
			cj += 1
		if 'BatchNormalization' in str(audioModel.layers[i]):
			audioModel.layers[i].set_weights([np.ones(means[bj].shape), np.zeros(means[bj].shape), means[bj], variances[bj]])
			bj += 1
			
			
	decoderModel = Decoder(faceModel.inputs, audioModel.inputs, faceModel.layers[-1].output, audioModel.layers[-1].output)
	
	
	
	
	for i in range(1, len(decoderModel.layers)):
		if 'conv2d_transpose' in str(decoderModel.layers[i]):
			if 'conv2d' in str(decoderModel.layers[i-6]):
				decoderModel.layers[i-6].set_weights([weights[cj], biases[cj]])
				cj += 1
			if 'BatchNormalization' in str(decoderModel.layers[i-4]):
				decoderModel.layers[i-4].set_weights([np.ones(means[bj].shape), np.zeros(means[bj].shape), means[bj], variances[bj]])
				bj += 1
			for j in range(i, len(decoderModel.layers)):
				if 'conv2d' in str(decoderModel.layers[j]):
					decoderModel.layers[j].set_weights([weights[cj], biases[cj]])
					cj += 1
				if 'BatchNormalization' in str(decoderModel.layers[j]):
					decoderModel.layers[j].set_weights([np.ones(means[bj].shape), np.zeros(means[bj].shape), means[bj], variances[bj]])
					bj += 1
			break
			
	for i in range(cj, len(weights)):
		print(weights[i].shape)'''
	
	
	print('Weights are initialized\n')
	
	print(cnt)
	




	return decoderModel
















def conv2d_block(x, filters, kernelSize, stride, pad, residual = False):
	y = Conv2D(filters, kernel_size = kernelSize, strides = stride, padding = pad)(x)
	y = BatchNormalization()(y)
	
	if residual == True:
		y += x
	
	y =  ReLU()(y)
	
	return y

def conv2d_transpose(x, filters, kernelSize, stride, pad, outpad = 0, Name = None):
	y = Conv2DTranspose(filters, kernel_size = kernelSize, strides = stride, padding = pad, output_padding = outpad, name = Name)(x)
	y = BatchNormalization()(y)
	y = ReLU()(y)
	
	return y
	
#face_encoder_output = []
	
'''def audioEncoder():
	
	#Audio Encoder
	input_audio = Input(shape=(80, 16, 1), name='AudioInput')
	
	audio_x1 = conv2d_block(input_audio, 32, (3,3), (1,1), 'same')
	audio_x2 = conv2d_block(audio_x1, 32, (3,3), (1,1), 'same', residual=True)
	audio_x2 = conv2d_block(audio_x2, 32, (3,3), (1,1), 'same', residual=True)

	audio_x3 = conv2d_block(audio_x2, 64, (3,3), (3,1), 'same')
	audio_x3 = conv2d_block(audio_x3, 64, (3,3), (1,1), 'same', residual=True)
	audio_x3 = conv2d_block(audio_x3, 64, (3,3), (1,1), 'same', residual=True)
	
	audio_x4 = conv2d_block(audio_x3, 128, (3,3), (3,3), 'same')
	audio_x4 = conv2d_block(audio_x4, 128, (3,3), (1,1), 'same', residual=True)
	audio_x4 = conv2d_block(audio_x4, 128, (3,3), (1,1), 'same', residual=True)
	
	audio_x5 = conv2d_block(audio_x4, 256, (3,3), (3,2), 'same')
	audio_x5 = conv2d_block(audio_x5, 256, (3,3), (1,1), 'same', residual=True)
	
	audio_x6 = conv2d_block(audio_x5, 512, (3,3), (1,1), 'valid')
	
	output_audio = conv2d_block(audio_x6, 512, (1,1), (1,1), 'valid')
	
	audio_model = Model(input_audio, output_audio)
	#audio_model.summary()
	
	
	return audio_model
'''	
	
'''def faceEncoder():

	#Face Encoder
	
	input_face = Input(shape=(96, 96, 6), name='FaceInput')
	
	face_x1 = conv2d_block(input_face, 16, (7,7), (1,1), 'same') #96,96
	face_encoder_output.append(face_x1)
	
	face_x2 = conv2d_block(face_x1, 32, (3,3), (2,2), 'same')
	face_x2 = conv2d_block(face_x2, 32, (3,3), (1,1), 'same', residual=True)
	face_x2 = conv2d_block(face_x2, 32, (3,3), (1,1), 'same', residual=True)
	face_encoder_output.append(face_x2)
	
	face_x3 = conv2d_block(face_x2, 64, (3,3), (2,2), 'same')
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	face_encoder_output.append(face_x3)
	
	face_x4 = conv2d_block(face_x3, 128, (3,3), (2,2), 'same')
	face_x4 = conv2d_block(face_x4, 128, (3,3), (1,1), 'same', residual=True)
	face_x4 = conv2d_block(face_x4, 128, (3,3), (1,1), 'same', residual=True)
	face_encoder_output.append(face_x4)
	

	face_x5 = conv2d_block(face_x4, 256, (3,3), (2,2), 'same')
	face_x5 = conv2d_block(face_x5, 256, (3,3), (1,1), 'same', residual=True)
	face_x5 = conv2d_block(face_x5, 256, (3,3), (1,1), 'same', residual=True)
	face_encoder_output.append(face_x5)
		
	face_x6 = conv2d_block(face_x5, 512, (3,3), (2,2), 'same')
	face_x6 = conv2d_block(face_x6, 512, (3,3), (1,1), 'same', residual=True)
	face_encoder_output.append(face_x6)
	
	face_x7 = conv2d_block(face_x6, 512, (3,3), (1,1), 'valid')
		
	output_face = conv2d_block(face_x7, 512, (1,1), (1,1), 'valid')
	
	
	face_model = Model(input_face, output_face)
	#face_model.summary()
	
	return face_model'''
	
def Decoder():

	'''faceModel = faceEncoder()
	audioModel = audioEncoder()
	
	audioOut = audioModel.layers[-1].output
	faceOut = faceModel.layers[-1].output'''
	
	
	
	#Face Encoder
	
	input_face = Input(shape=(96, 96, 6), name='FaceInput')
	
	face_x1 = conv2d_block(input_face, 16, (7,7), (1,1), 'same') #96,96
	#face_encoder_output.append(face_x1)
	
	face_x2 = conv2d_block(face_x1, 32, (3,3), (2,2), 'same')
	face_x2 = conv2d_block(face_x2, 32, (3,3), (1,1), 'same', residual=True)
	face_x2 = conv2d_block(face_x2, 32, (3,3), (1,1), 'same', residual=True)
	#face_encoder_output.append(face_x2)
	
	face_x3 = conv2d_block(face_x2, 64, (3,3), (2,2), 'same')
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	face_x3 = conv2d_block(face_x3, 64, (3,3), (1,1), 'same', residual=True)
	#face_encoder_output.append(face_x3)
	
	face_x4 = conv2d_block(face_x3, 128, (3,3), (2,2), 'same')
	face_x4 = conv2d_block(face_x4, 128, (3,3), (1,1), 'same', residual=True)
	face_x4 = conv2d_block(face_x4, 128, (3,3), (1,1), 'same', residual=True)
	#face_encoder_output.append(face_x4)
	

	face_x5 = conv2d_block(face_x4, 256, (3,3), (2,2), 'same')
	face_x5 = conv2d_block(face_x5, 256, (3,3), (1,1), 'same', residual=True)
	face_x5 = conv2d_block(face_x5, 256, (3,3), (1,1), 'same', residual=True)
	#face_encoder_output.append(face_x5)
		
	face_x6 = conv2d_block(face_x5, 512, (3,3), (2,2), 'same')
	face_x6 = conv2d_block(face_x6, 512, (3,3), (1,1), 'same', residual=True)
	#face_encoder_output.append(face_x6)
	
	face_x7 = conv2d_block(face_x6, 512, (3,3), (1,1), 'valid')
		
	output_face = conv2d_block(face_x7, 512, (1,1), (1,1), 'valid')
	
	
	
	
	#Audio Encoder
	input_audio = Input(shape=(80, 16, 1), name='AudioInput')
	
	audio_x1 = conv2d_block(input_audio, 32, (3,3), (1,1), 'same')
	audio_x2 = conv2d_block(audio_x1, 32, (3,3), (1,1), 'same', residual=True)
	audio_x2 = conv2d_block(audio_x2, 32, (3,3), (1,1), 'same', residual=True)

	audio_x3 = conv2d_block(audio_x2, 64, (3,3), (3,1), 'same')
	audio_x3 = conv2d_block(audio_x3, 64, (3,3), (1,1), 'same', residual=True)
	audio_x3 = conv2d_block(audio_x3, 64, (3,3), (1,1), 'same', residual=True)
	
	audio_x4 = conv2d_block(audio_x3, 128, (3,3), (3,3), 'same')
	audio_x4 = conv2d_block(audio_x4, 128, (3,3), (1,1), 'same', residual=True)
	audio_x4 = conv2d_block(audio_x4, 128, (3,3), (1,1), 'same', residual=True)
	
	audio_x5 = conv2d_block(audio_x4, 256, (3,3), (3,2), 'same')
	audio_x5 = conv2d_block(audio_x5, 256, (3,3), (1,1), 'same', residual=True)
	
	audio_x6 = conv2d_block(audio_x5, 512, (3,3), (1,1), 'valid')
	
	output_audio = conv2d_block(audio_x6, 512, (1,1), (1,1), 'valid')
	
	
	#face decoder 
	dec_x0 = conv2d_block(output_audio, 512, (1,1), (1,1), 'valid')

	embeddings = Concatenate(axis=3)([output_face, dec_x0])

	dec_x1 = conv2d_transpose(embeddings, 512, (3,3), (1,1), 'valid', 0, 'Decoder_In') #3,3
	dec_x2 = conv2d_block(dec_x1, 512, (3,3), (1,1), 'same', residual=True)
	dec_x2 = Concatenate(axis=3)([face_x6, dec_x2])
	#dec_x2 = Concatenate(axis=3)([face_encoder_output[5], dec_x2])
	
	
	dec_x3 = conv2d_transpose(dec_x2, 512, (3,3), (2,2), 'same', 1)
	dec_x3 = conv2d_block(dec_x3, 512, (3,3), (1,1), 'same', residual=True)
	dec_x3 = conv2d_block(dec_x3, 512, (3,3), (1,1), 'same', residual=True)
	dec_x3 = Concatenate(axis=3)([face_x5, dec_x3]) #6,6
	#dec_x3 = Concatenate(axis=3)([face_encoder_output[4], dec_x3])
	
	
	dec_x4 = conv2d_transpose(dec_x3, 384, (3,3), (2,2), 'same', 1)
	dec_x4 = conv2d_block(dec_x4, 384, (3,3), (1,1), 'same', residual=True)
	dec_x4 = conv2d_block(dec_x4, 384, (3,3), (1,1), 'same', residual=True)
	dec_x4 = Concatenate(axis=3)([face_x4, dec_x4])
	#dec_x4 = Concatenate(axis=3)([face_encoder_output[3], dec_x4]) #12,12
	
	
	dec_x5 = conv2d_transpose(dec_x4, 256, (3,3), (2,2), 'same', 1)
	dec_x5 = conv2d_block(dec_x5, 256, (3,3), (1,1), 'same', residual=True)
	dec_x5 = conv2d_block(dec_x5, 256, (3,3), (1,1), 'same', residual=True)
	dec_x5 = Concatenate(axis=3)([face_x3, dec_x5])
	#dec_x5 = Concatenate(axis=3)([face_encoder_output[2], dec_x5]) #24,24
	
	
	dec_x6 = conv2d_transpose(dec_x5, 128, (3,3), (2,2), 'same', 1)
	dec_x6 = conv2d_block(dec_x6, 128, (3,3), (1,1), 'same', residual=True)
	dec_x6 = conv2d_block(dec_x6, 128, (3,3), (1,1), 'same', residual=True)
	dec_x6 = Concatenate(axis=3)([face_x2, dec_x6])
	#dec_x6 = Concatenate(axis=3)([face_encoder_output[1], dec_x6]) #48,48
	
	
	dec_x7 = conv2d_transpose(dec_x6, 64, (3,3), (2,2), 'same', 1)
	dec_x7 = conv2d_block(dec_x7, 64, (3,3), (1,1), 'same', residual=True)
	dec_x7 = conv2d_block(dec_x7, 64, (3,3), (1,1), 'same', residual=True)
	dec_x7 = Concatenate(axis=3)([face_x1, dec_x7])
	#dec_x7 = Concatenate(axis=3)([face_encoder_output[0], dec_x7]) #96,96
	
	
	#output
	dec_x8 = conv2d_block(dec_x7, 32, (3,3), (1,1), 'same')

	dec_x9 = Conv2D(3, kernel_size=(1,1), strides=(1,1), padding='valid')(dec_x8)
	dec_output = Activation('sigmoid')(dec_x9)
	
	
	decoder_model = Model([input_audio, input_face], dec_output)
	decoder_model.summary()
	
	return decoder_model	
	
	
def main():

	decoderModel = loading_model_weights()
	
	#decoderModel = Decoder();
	
	
	#faceModel, audioModel, decoderModel = loading_model_weights()
	
	'''audioModel.save_weights('checkpoint_gen/audio/audio.ckpt')
	faceModel.save_weights('checkpoint_gen/face/face.ckpt')'''
	
	'''decoderModel.save_weights('checkpoint_gen/generator/generator.ckpt')
	
	print("Model saved")'''
	

	
	
	#decoder_weights = 'checkpoint_gen/generator/generator.ckpt'
	
	#decoderModel = Decoder()
	#decoderModel.load_weights(decoder_weights)
	
	
	#print(len(decoderModel.layers))
	
	
	
	
	'''for layer in decoderModel.layers:
		print(layer)
    	#weights = layer.get_weights()
	print(len(decoderModel.layers))'''
	
	face_seq, audio_seq = getitem()
	
	generated_img = decoderModel.predict([audio_seq, face_seq])
	
	print(K.eval(decoderModel.layers[45].output))
	
	print("Shoumik ", generated_img.shape)
	
	
	
	
	'''plt.figure(figsize = (20, 20))
	for i in range(5):
		plt.subplot(3, 3, i + 1)
		plt.axis('off')
		plt.imshow(generated_img[i])	
	plt.show()'''

		
	
		
	
if __name__ == '__main__':
	main()
	
