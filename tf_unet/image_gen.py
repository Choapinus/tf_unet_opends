# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
import glob
import numpy as np
from tf_unet.image_util import BaseDataProvider, ImageDataProvider

class GrayScaleDataProvider(BaseDataProvider):
	channels = 1
	n_class = 2
	
	def __init__(self, nx, ny, **kwargs):
		super(GrayScaleDataProvider, self).__init__()
		self.nx = nx
		self.ny = ny
		self.kwargs = kwargs
		rect = kwargs.get("rectangles", False)
		if rect:
			self.n_class=3
		
	def _next_data(self):
		return create_image_and_label(self.nx, self.ny, **self.kwargs)

class RgbDataProvider(BaseDataProvider):
	channels = 3
	n_class = 2
	
	def __init__(self, nx, ny, **kwargs):
		super(RgbDataProvider, self).__init__()
		self.nx = nx
		self.ny = ny
		self.kwargs = kwargs
		rect = kwargs.get("rectangles", False)
		if rect:
			self.n_class=3

		
	def _next_data(self):
		data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
		return to_rgb(data), label

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

	def _process_data(self, data):
		return data/255.0

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


def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
	
	
	image = np.ones((nx, ny, 1))
	label = np.zeros((nx, ny, 3), dtype=np.bool)
	mask = np.zeros((nx, ny), dtype=np.bool)
	for _ in range(cnt):
		a = np.random.randint(border, nx-border)
		b = np.random.randint(border, ny-border)
		r = np.random.randint(r_min, r_max)
		h = np.random.randint(1,255)

		y,x = np.ogrid[-a:nx-a, -b:ny-b]
		m = x*x + y*y <= r*r
		mask = np.logical_or(mask, m)

		image[m] = h

	label[mask, 1] = 1
	
	if rectangles:
		mask = np.zeros((nx, ny), dtype=np.bool)
		for _ in range(cnt//2):
			a = np.random.randint(nx)
			b = np.random.randint(ny)
			r =  np.random.randint(r_min, r_max)
			h = np.random.randint(1,255)
	
			m = np.zeros((nx, ny), dtype=np.bool)
			m[a:a+r, b:b+r] = True
			mask = np.logical_or(mask, m)
			image[m] = h
			
		label[mask, 2] = 1
		
		label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
	
	image += np.random.normal(scale=sigma, size=image.shape)
	image -= np.amin(image)
	image /= np.amax(image)
	
	if rectangles:
		return image, label
	else:
		return image, label[..., 1]




def to_rgb(img):
	img = img.reshape(img.shape[0], img.shape[1])
	img[np.isnan(img)] = 0
	img -= np.amin(img)
	img /= np.amax(img)
	blue = np.clip(4*(0.75-img), 0, 1)
	red  = np.clip(4*(img-0.25), 0, 1)
	green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
	rgb = np.stack((red, green, blue), axis=2)
	return rgb
