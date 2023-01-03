from os.path import dirname, join, basename, isfile
#from tqdm import tqdm

#from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob
import tensorflow as tf

import os, random, cv2, argparse
from hparams import hparams, get_image_list

syncnet_T = 5
syncnet_mel_step_size = 16

def get_frame_id(frame):
		return int(basename(frame).split('.')[0])
	
def get_window(start_frame):
        start_id = get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + 5):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

def crop_audio_window(spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = start_frame
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + 16

        return spec[start_idx : end_idx, :]
 
 
def read_window(window_fnames):
	if window_fnames is None: return None
	window = []
	for fname in window_fnames:
		img = cv2.imread(fname)
		if img is None:
			return None
		try:
			img = cv2.resize(img, (hparams.img_size, hparams.img_size))
		except Exception as e:
			return None

		window.append(img)

	return window
		
		
def get_segmented_mels(spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = get_frame_id(start_frame)
        #if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = crop_audio_window(spec, i)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels
        
        
        
def prepare_window(window):
        # 3 x T x H x W
		x = np.asarray(window) / 255.
		#x = np.transpose(x, (3, 0, 1, 2))

		return x
        

        

def getitem():

	vidname = 'dataset'
				
	#while 1:
	img_names = list(glob(join(vidname, '*.jpg')))
	
	img_dirs = []
    
	for i in range(0,10):
		for j in img_names:
			if i == get_frame_id(j):
				img_dirs.append(j)
				break
				
	img_name = img_dirs[0:5]
	wrong_img_name = img_dirs[-5:]
	
	window_fnames = get_window(img_name[0])
	wrong_window_fnames = get_window(wrong_img_name[0])
	
	window = read_window(window_fnames)
	wrong_window = read_window(wrong_window_fnames)
	
	
	try:
		wavpath = join(vidname, "audio.wav")
		wav = audio.load_wav(wavpath, hparams.sample_rate)

		orig_mel = audio.melspectrogram(wav).T
	except Exception as e:
		print(e)
		
		
	indiv_mels = get_segmented_mels(orig_mel.copy(), img_name[0])
	
	window = prepare_window(window)
	wrong_window = prepare_window(wrong_window)
	
	window[:, window.shape[2]//2:, :] = 0.
	
	x = np.concatenate([window, wrong_window], axis=3)
	
	
	x = tf.convert_to_tensor(x, dtype='float32')
	indiv_mels = tf.convert_to_tensor(indiv_mels, dtype='float32')
	
	
	indiv_mels = tf.expand_dims(indiv_mels, axis=3)
	
	
	return x, indiv_mels
		
	
	

