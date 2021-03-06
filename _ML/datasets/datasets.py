'''****************************************************************************
 * datasets.py: Dataset Loading, Saving, Transformation, and Visualization
 ******************************************************************************
 * v0.1 - 01.03.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************'''


import cv2
import h5py
import os

import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf

from glob    import glob
from natsort import natsorted
from time    import time

from core.Hexnet import Hexsamp_s2h, Sqsamp_s2s
from misc.misc   import Hexnet_print


def create_dataset_h5(
	dataset,
	train_classes,
	train_data,
	train_filenames,
	train_labels,
	test_classes,
	test_data,
	test_filenames,
	test_labels):

	with h5py.File(dataset, 'w') as h5py_file:
		h5py_file.create_dataset('train_classes',   data=train_classes)
		h5py_file.create_dataset('train_data',      data=train_data)
		h5py_file.create_dataset('train_filenames', data=train_filenames)
		h5py_file.create_dataset('train_labels',    data=train_labels)
		h5py_file.create_dataset('test_classes',    data=test_classes)
		h5py_file.create_dataset('test_data',       data=test_data)
		h5py_file.create_dataset('test_filenames',  data=test_filenames)
		h5py_file.create_dataset('test_labels',     data=test_labels)


def load_dataset(dataset, create_h5=True, verbosity_level=2):
	Hexnet_print(f'Loading dataset {dataset}')

	train_classes   = []
	train_data      = []
	train_filenames = []
	train_labels    = []
	test_classes    = []
	test_data       = []
	test_filenames  = []
	test_labels     = []

	if os.path.isfile(dataset) and dataset.endswith('.h5'):
		start_time = time()

		with h5py.File(dataset, 'r') as h5py_file:
			train_classes   = np.array(h5py_file['train_classes'])
			train_data      = np.array(h5py_file['train_data'])
			train_filenames = np.array(h5py_file['train_filenames'])
			train_labels    = np.array(h5py_file['train_labels'])
			test_classes    = np.array(h5py_file['test_classes'])
			test_data       = np.array(h5py_file['test_data'])
			test_filenames  = np.array(h5py_file['test_filenames'])
			test_labels     = np.array(h5py_file['test_labels'])

		time_diff = time() - start_time

		Hexnet_print(f'Loaded dataset {dataset} in {time_diff:.3f} seconds')
	else:
		start_time = time()

		for dataset_set in natsorted(glob(os.path.join(dataset, '*'))):
			current_set = os.path.basename(dataset_set)

			if verbosity_level >= 1:
				Hexnet_print(f'\t> current_set={current_set}')

			for set_class in natsorted(glob(os.path.join(dataset_set, '*'))):
				current_class = os.path.basename(set_class)

				if verbosity_level >= 2:
					Hexnet_print(f'\t\t> current_class={current_class}')

				if 'train' in current_set:
					train_classes.append(current_class)
				elif 'test' in current_set:
					test_classes.append(current_class)

				for class_image in natsorted(glob(os.path.join(set_class, '*'))):
					current_image = os.path.basename(class_image)

					if verbosity_level >= 3:
						Hexnet_print(f'\t\t\t> current_image={current_image}')

					if 'train' in current_set:
						train_data.append(cv2.imread(class_image, cv2.IMREAD_COLOR))
						train_filenames.append(current_image)
						train_labels.append(current_class)
					elif 'test' in current_set:
						test_data.append(cv2.imread(class_image, cv2.IMREAD_COLOR))
						test_filenames.append(current_image)
						test_labels.append(current_class)

		time_diff = time() - start_time

		Hexnet_print(f'Loaded dataset {dataset} in {time_diff:.3f} seconds')

		if create_h5:
			dataset         = f'{dataset}.h5'
			train_classes   = np.array(train_classes,   dtype='string_')
			train_data      = np.array(train_data)
			train_filenames = np.array(train_filenames, dtype='string_')
			train_labels    = np.array(train_labels,    dtype='string_')
			test_classes    = np.array(test_classes,    dtype='string_')
			test_data       = np.array(test_data)
			test_filenames  = np.array(test_filenames,  dtype='string_')
			test_labels     = np.array(test_labels,     dtype='string_')

			create_dataset_h5(
				dataset,
				train_classes,
				train_data,
				train_filenames,
				train_labels,
				test_classes,
				test_data,
				test_filenames,
				test_labels)

	return ((train_classes, train_data, train_filenames, train_labels), (test_classes, test_data, test_filenames, test_labels))


def transform_dataset(
	dataset,
	output_dir,
	mode            = 's2h',
	rad_o           =  1.0,
	width           = 64,
	height          = None,
	method          =  0,
	verbosity_level =  2):

	if os.path.exists(output_dir):
		Hexnet_print(f'Dataset {output_dir} exists already (skipping transformation)')
		return

	if os.path.isfile(f'{output_dir}.h5'):
		Hexnet_print(f'Dataset {output_dir}.h5 exists already (skipping transformation)')
		return

	Hexnet_print(f'Transforming dataset {dataset} with mode {mode} to {output_dir}')

	os.makedirs(output_dir, exist_ok=True)

	for dataset_set in natsorted(glob(os.path.join(dataset, '*'))):
		current_set = os.path.basename(dataset_set)

		if verbosity_level >= 1:
			print(f'\t> current_set={current_set}')

		output_dir_current_set = os.path.join(output_dir, current_set)
		os.makedirs(output_dir_current_set, exist_ok=True)

		for set_class in natsorted(glob(os.path.join(dataset_set, '*'))):
			current_class = os.path.basename(set_class)

			if verbosity_level >= 2:
				print(f'\t\t> current_class={current_class}')

			output_dir_current_class = os.path.join(output_dir_current_set, current_class)
			os.makedirs(output_dir_current_class, exist_ok=True)

			if mode == 's2h':
				Hexsamp_s2h(
					filename_s         = os.path.join(set_class, '*'),
					output_dir         = output_dir_current_class,
					rad_o              = rad_o,
					method             = method,
					increase_verbosity = True if verbosity_level >= 3 else False)
			else:
				Sqsamp_s2s(
					filename_s         = os.path.join(set_class, '*'),
					output_dir         = output_dir_current_class,
					width              = width,
					height             = height,
					method             = method,
					increase_verbosity = True if verbosity_level >= 3 else False)


def resize_dataset(dataset_s, resize_string, method='nearest'):
	# HxW
	resize_size = resize_string.split('x')
	resize_H    = int(resize_size[0])
	resize_W    = int(resize_size[1])
	target_size = (resize_H, resize_W)

	if not type(dataset_s) is list:
		dataset_s = list(dataset_s)

	dataset_s = [tf.image.resize(dataset, size=target_size, method=method).numpy() for dataset in dataset_s]

	return dataset_s


def crop_dataset(dataset_s, crop_string):
	# HxW+Y+X
	crop_offset = crop_string.split('+')
	crop_size   = crop_offset[0].split('x')
	crop_offset = crop_offset[1:]

	if not type(dataset_s) is list:
		dataset_s = list(dataset_s)

	if not '+' in crop_string:
		crop_Y = 0
		crop_X = 0
	else:
		crop_Y = int(crop_offset[0])
		crop_X = int(crop_offset[1])

	if not 'x' in crop_string:
		crop_H = dataset_s[0].shape[0] - crop_Y
		crop_W = dataset_s[0].shape[1] - crop_X
	else:
		crop_H = int(crop_size[0])
		crop_W = int(crop_size[1])

	slice_H = slice(crop_Y, crop_Y + crop_H)
	slice_W = slice(crop_X, crop_X + crop_W)

	dataset_s = [dataset[:,slice_H,slice_W,:] for dataset in dataset_s]

	return dataset_s


def show_dataset_classes(
	train_classes,
	train_data,
	train_labels,
	test_classes,
	test_data,
	test_labels,
	max_images_per_class   =  1,
	max_classes_to_display = 10):

	nrows     = 2 * max_images_per_class
	ncols     = min(len(train_classes), max_classes_to_display)
	figsize_2 = max_images_per_class * ncols
	index     = 1

	plt.figure('Dataset classes')
	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	for class_counter, train_class in enumerate(train_classes):
		if class_counter == max_classes_to_display:
			break

		class_label_indices = np.where(train_labels == train_class)[0]

		for image_counter in range(max_images_per_class):
			plt.subplot(nrows, ncols, index)
			plt.title(f'train image {index}\n(class {train_class})')
			plt.imshow(train_data[class_label_indices[image_counter]])

			index += 1

	index = figsize_2 + 1

	for class_counter, test_class in enumerate(test_classes):
		if class_counter == max_classes_to_display:
			break

		class_label_indices = np.where(test_labels == test_class)[0]

		for image_counter in range(max_images_per_class):
			plt.subplot(nrows, ncols, index)
			plt.title(f'test image {index - figsize_2}\n(class {test_class})')
			plt.imshow(test_data[class_label_indices[image_counter]])

			index += 1

	plt.show()

	plt.close()


