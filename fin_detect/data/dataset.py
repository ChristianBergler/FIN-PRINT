"""
Module: dataset.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

"""
Code from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/datasets.py
License: GNU General Public License v3.0
Access Data: 06.06.2020, Last Access Date: 25.01.2022
Changes: Modified by Christian Bergler, Alexander Gebhard (continuous since 06.06.2020)
"""

import os
import glob
import math
import random
import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset

from data.augmentation import horizontal_flip


"""
Padding and extending smaller image dimension (width versus height) to get a squared image
"""
def pad_to_square(img, pad_value):
	c, h, w = img.shape
	dim_diff = np.abs(h - w)
	pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
	pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
	img = F.pad(img, pad, "constant", value=pad_value)
	return img, pad

"""
Resize a given squared image to the chosen image target size
"""
def resize(image, size):
	image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
	return image

"""
Resize a given squared image to a random image target size within a min/max range according to the chosen image target size
"""
def random_resize(images, min_size=288, max_size=448):
	new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
	images = F.interpolate(images, size=new_size, mode="nearest")
	return images

"""
Compare two given images if the artist (photographer) and date is both identical
"""
def date_and_artist_are_equal(fp, previous_fp):
	fn_split = fp.replace('/', '\\').split('\\')[-1].split('_')
	fn_date, fn_artist = fn_split[1], fn_split[2]
	prev_fn_split = previous_fp.replace('/', '\\').split('\\')[-1].split('_')
	prev_fn_date, prev_fn_artist = prev_fn_split[1], prev_fn_split[2]

	return fn_date == prev_fn_date and fn_artist == prev_fn_artist

"""
Creates the datasets (csv file), including the previously generated image files
"""
def generate_csv_files(data_split_dir: str, train: list, val: list, test: list):
	train_path = os.path.join(data_split_dir, "train.txt")
	valid_path = os.path.join(data_split_dir, "valid.txt")
	test_path = os.path.join(data_split_dir, "test.txt")

	train_file = open(train_path, "w")
	valid_file = open(valid_path, "w")
	test_file = open(test_path, "w")

	for f in train:
		train_file.write(f+"\n")
	for f in val:
		valid_file.write(f+"\n")
	for f in test:
		test_file.write(f+"\n")

	train_file.close()
	valid_file.close()
	test_file.close()

	return train_path, valid_path, test_path

"""
Generation of training, validation, and test dataset (e.g. 70% train, 15% validation, and 15% test) while offering an option
for allowing no images of the same photographer and date within the validation and/or test set (except training). By default
this option is not used, because it might remove many of the existing data.
"""
def create_data_split(path, logger, log_dir, train_split: float, photographer_and_date_constraint=False):

	unused = []
	all_filepaths = []

	ds = open(f"{log_dir}/DATASET.log", "w")

	for (dirpath, dirnames, filenames) in os.walk(path):
		for fn in sorted(filenames):
			if not fn.lower().endswith(".jpg") and not fn.lower().endswith(".JPG") and not fn.lower().endswith(".png"):
				continue
			fp = os.path.abspath(os.path.join(dirpath, fn))
			all_filepaths.append(fp)

	n = len(all_filepaths)
	n_train = math.floor(n * train_split)
	n_valid = math.floor((n - n_train) / 2)
	n_test = (n-(n_valid+n_train))

	ds.write(f"\nOriginal Image Number: {n}\nTraining Samples: {n_train}\nValidation Samples: {n_valid}\nTest Samples: {n_test}\nSum Data Split: {n_train+n_test+n_valid}\n")

	random.shuffle(all_filepaths)
	random.shuffle(all_filepaths)
	random.shuffle(all_filepaths)

	train_fp, valid_fp, test_fp = all_filepaths[:n_train], all_filepaths[n_train: n_train + n_valid], all_filepaths[n_train+n_valid:]

	logger.info("Files before applying the interval rule!")
	logger.info("Original Image Number: " + str(n))
	logger.info("Number of Training Samples: " + str(n_train))
	logger.info("Training Samples: " + str(train_fp))
	logger.info("Number of Validation Samples: " + str(n_valid))
	logger.info("Validation Samples: " + str(valid_fp))
	logger.info("Number of Test Samples: " + str(n_test))
	logger.info("Test Samples: " + str(test_fp))
	logger.info("Sum Data Split: " + str(n_train + n_test + n_valid))

	fp_old = None

	if photographer_and_date_constraint:
		logger.info("Photographer/Date constraint is set for data partitioning - original number of images compared to number of images after the data partitioning might differ!")
		if valid_fp:
			for fp in valid_fp[:]:
				if fp_old is not None and date_and_artist_are_equal(fp, fp_old):
					unused.append(fp)
					valid_fp.remove(fp)
				fp_old = fp
		else:
			pass

		fp_old = None

		if test_fp:
			for fp in test_fp[:]:
				if fp_old is not None and date_and_artist_are_equal(fp, fp_old):
					unused.append(fp)
					test_fp.remove(fp)
				fp_old = fp
		else:
			pass

		n_train = len(train_fp)
		n_valid = len(valid_fp)
		n_test = len(test_fp)

		if len(train_fp) == 0 or len(valid_fp) == 0 or len(test_fp) == 0:
			logger.info("Number Training Files: " + str(len(train_fp)))
			logger.info("Number Validation Files: " + str(len(valid_fp)))
			logger.info("Number Testing Files: " + str(len(test_fp)))
			raise ValueError("At least one of the datasets is empty. Please, either add more files to the single distributions or check the time interval rule whether too much is be removed due to that rule. Abort...")

		logger.info("\n\nFiles after applying the photographer/date constraint on the valid and test partition, except for the training dataset!")
		logger.info("Original Image Number: " + str(n))
		logger.info("Number of Training Samples: " + str(n_train))
		logger.info("Training Samples: " + str(train_fp))
		logger.info("Number of Validation Samples: " + str(n_valid))
		logger.info("Validation Samples: " + str(valid_fp))
		logger.info("Number of Test Samples: " + str(n_test))
		logger.info("Test Samples: " + str(test_fp))
		logger.info("Sum Data Split: " + str(n_train + n_test + n_valid))
	else:
		logger.info("Photographer/Date constraint not set for data partitioning - original number of images compared to number of images after the data partitioning have to be equal!")

	train_all = train_fp
	valid_all = valid_fp
	test_all = test_fp

	random.shuffle(train_all)
	random.shuffle(valid_all)
	random.shuffle(test_all)

	train_all_n, valid_all_n, test_all_n = len(train_all), len(valid_all), len(test_all)
	logger.info("\n\n--- Main sets including all classes ---")
	logger.info("Original Image Number: " + str(train_all_n + valid_all_n + test_all_n))
	logger.info("Training Samples: " + str(train_all_n))
	logger.info("Validation Samples: " + str(valid_all_n))
	logger.info("Test Samples: " + str(test_all_n))
	logger.info("Sum Data Split: " + str(train_all_n + valid_all_n + test_all_n) + "\n")

	ds.write(f"\n\n--- Main sets including all data samples ---\nOriginal Image Number: {train_all_n + valid_all_n + test_all_n}\nTraining Samples: {train_all_n}\n"
			 f"Validation Samples: {valid_all_n}\nTest Samples: {test_all_n}\nSum Data Split: {train_all_n + valid_all_n + test_all_n}\n")

	logger.info("Train-Set: " + str(train_all))
	logger.info("Val-Set: " + str(valid_all))
	logger.info("Test-Set: " + str(test_all))

	ds.write(f"\n\nTrain-Set: {train_all}\n\nVal-Set: {valid_all}\n\nTest-Set: {test_all}\n")

	ds.close()

	logger.info("\n\n------------ ALL files which were ignored by the interval rule ---------------")
	logger.info("Unused files due to the photographer/date constraint")
	logger.info("Number of unused files (validation and test set): " + str(len(unused)))
	logger.info("Unused files: " + str(unused))
	logger.info("---------------------------")

	return train_all, valid_all, test_all






"""
Dataset taking a given folder of images, padding them to a square resolution, resizing, and use them for prediction with a given model
"""
class ImageFolder(Dataset):

	def __init__(self, folder_path, img_size=416):
		self.files = sorted(glob.glob("%s/*.*" % folder_path))
		self.img_size = img_size

	def __getitem__(self, index):
		img_path = self.files[index % len(self.files)]
		try:
			img = transforms.ToTensor()(Image.open(img_path))
		except:
			raise Exception("Image File: " + str(img_path) + " could not be opened")
		img, _ = pad_to_square(img, 0)
		img = resize(img, self.img_size)
		return img_path, img

	def __len__(self):
		return len(self.files)

"""
Dataset for training, validation, and testing, taking images according to the data split files within the data config file
padding and resizing them, reading and resizing ground truth bounding box label information, perform augmentation for
train (not validation and test), as well as returning the given batch, images paths, next to the ground truth targets
"""
class Dataset(Dataset):

	def __init__(self, data_file_list, split, img_size=416, max_obj=100, augment=True, multiscale=True, normalized_labels=True):

		self.split = split
		self.augment = augment
		self.img_size = img_size
		self.max_objects = max_obj
		self.multiscale = multiscale
		self.img_files = data_file_list
		self.normalized_labels = normalized_labels

		self.batch_count = 0
		self.min_size = self.img_size - 3 * 32
		self.max_size = self.img_size + 3 * 32

		self.label_files = [
			path.replace("images", "detections").replace(".png", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
			for path in self.img_files
		]

	def __getitem__(self, index):

		img_path = self.img_files[index % len(self.img_files)].rstrip()
		img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

		if len(img.shape) != 3:
			img = img.unsqueeze(0)
			img = img.expand((3, img.shape[1:]))

		_, h, w = img.shape
		h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

		img, pad = pad_to_square(img, 0)
		_, padded_h, padded_w = img.shape

		label_path = self.label_files[index % len(self.img_files)].rstrip()
		targets = None
		if os.path.exists(label_path):
			boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
			# Extract coordinates for unpadded + unscaled image
			x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
			y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
			x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
			y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
			# Adjust for added padding
			x1 += pad[0]
			y1 += pad[2]
			x2 += pad[1]
			y2 += pad[3]
			# Returns (x, y, w, h)
			boxes[:, 1] = ((x1 + x2) / 2) / padded_w
			boxes[:, 2] = ((y1 + y2) / 2) / padded_h
			boxes[:, 3] *= w_factor / padded_w
			boxes[:, 4] *= h_factor / padded_h

			targets = torch.zeros((len(boxes), 6))
			targets[:, 1:] = boxes

		if self.augment:
			if np.random.random() < 0.5:
				img, targets = horizontal_flip(img, targets)

		return img_path, img, targets

	def collate_fn(self, batch):
		paths, imgs, targets = list(zip(*batch))
		targets = [boxes for boxes in targets if boxes is not None]
		for i, boxes in enumerate(targets):
			boxes[:, 0] = i
		targets = torch.cat(targets, 0)
		# Selects new image size every tenth batch otherwise constant image size (e.g. 416x416)
		# only if multiscale option is set
		if self.multiscale and self.batch_count % 10 == 0:
			self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
		imgs = torch.stack([resize(img, self.img_size) for img in imgs])
		self.batch_count += 1
		return paths, imgs, targets

	def __len__(self):
		return len(self.img_files)
