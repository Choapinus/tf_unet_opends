from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
import glob
import numpy as np
from tf_unet.image_util import BaseDataProvider

class OpendsDataProvider(BaseDataProvider):
	def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".png", mask_suffix='_mask.tif', shuffle_data=True):
		super(OpendsDataProvider, self).__init__(a_min, a_max)
		self.data_suffix = data_suffix
		self.mask_suffix = mask_suffix
		self.file_idx = -1
		self.shuffle_data = shuffle_data

		self.data_files = self._find_data_files(search_path)

		if self.shuffle_data:
			np.random.shuffle(self.data_files)

		assert len(self.data_files) > 0, "No training files"
		print("Number of files used: %s" % len(self.data_files))

		image_path = self.data_files[0]
		label_path = image_path.replace(self.data_suffix, self.mask_suffix)
		img = self._load_file(image_path)
		mask = self._load_file(label_path)
		self.channels = img.shape[-1]
		self.n_class = 4

		print("Number of channels: %s"%self.channels)
		print("Number of classes: %s"%self.n_class)

	def _find_data_files(self, search_path):
		all_files = glob.glob(search_path)
		return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

	def _load_file(self, path):
		im = cv2.imread(path)
		return im

	def _cylce_file(self):
		self.file_idx += 1
		if self.file_idx >= len(self.data_files):
			self.file_idx = 0
			if self.shuffle_data:
				np.random.shuffle(self.data_files)

	def _next_data(self):
		self._cylce_file()
		image_name = self.data_files[self.file_idx]
		label_name = image_name.replace(self.data_suffix, self.mask_suffix)

		img = self._load_file(image_name)
		label = self._load_file(label_name)

		return img, label

	# def _process_data(self, data):
	# 	return data/255.0

	def _load_data_and_label(self):
		data, label = self._next_data()

		train_data = self._process_data(data)
		labels = self._process_labels(label)

		train_data, labels = self._post_process(train_data, labels)

		nx = train_data.shape[1]
		ny = train_data.shape[0]

		return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.channels)
	
	def __call__(self, n):
		train_data, labels = self._load_data_and_label()
		nx = train_data.shape[1]
		ny = train_data.shape[2]

		X = np.zeros((n, nx, ny, self.channels), dtype=np.float64)
		Y = np.zeros((n, nx, ny, self.channels), dtype=np.uint8)

		X[0] = train_data
		Y[0] = labels

		for i in range(1, n):
			train_data, labels = self._load_data_and_label()
			X[i] = train_data
			Y[i] = labels

		return X, Y