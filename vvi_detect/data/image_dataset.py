"""
Module: image_dataset.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import json
import math
import random
import datetime
import argparse
import collections

import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from data.augmentation import augment

parser = argparse.ArgumentParser()

DefaultImgDatasetOps = {
	"max_aug": 5,
	 "img_size": 512,
	"interval": 30,
	"across_split": True,
	"grayscale": False,
	"augmentation": True,
	"create_datasets": False,
}

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
Re-Shuffle two given datasets in case the date and artist (photographer) constraint is not fulfilled, while also considering
the "interval rule" (consecutive frames must be further away from each other than the specified interval rule)
"""
def sort_new_by_interval(set1, set2, interval=5):
	fp_old = set1[-1]
	fn_old = fp_old.replace("/", "\\").split("\\")[-1]
	for fp in set2[:]:
		fn = fp.replace("/", "\\").split("\\")[-1]
		if not check_time_interval(img1_fn=fn_old, img2_fn=fn, interval=interval):
			set1.append(fp)
			set2.remove(fp)
		fp_old = fp
		fn_old = fn
	return set1, set2

"""
Checks whether two given input images fulfill the time interval criteria or not, meaning frames must be further away 
from each other than the specified interval rule
"""
def check_time_interval(img1_fn, img2_fn, interval=5):

	img1_date, img1_time = img1_fn.split('.')[0].split('_')[-2:]
	if img1_time == 'None':
		return True

	year1, month1, day1 = img1_date.strip().split('-')
	hour1, min1, sec1 = img1_time.strip().split('-')
	img1_datetime = datetime.datetime(int(year1), int(month1), int(day1), int(hour1), int(min1), int(sec1))

	img2_date, img2_time = img2_fn.split('.')[0].split('_')[-2:]
	if img2_time == 'None':
		return True

	year2, month2, day2 = img2_date.strip().split('-')
	hour2, min2, sec2 = img2_time.strip().split('-')

	img2_datetime = datetime.datetime(int(year2), int(month2), int(day2), int(hour2), int(min2), int(sec2))

	diff_seconds = abs((img1_datetime - img2_datetime).total_seconds())

	return diff_seconds > interval

"""
Generation of training, validation, and test dataset for each class/category based on the given train split 
(e.g. 70% train, 15% validation, and 15% test) while applying the interval rule/constraint to the validation and 
test (not training). Split all sorted (after date and photographer) class-/category-specific filenames into the three 
partitions and in case of validation and test verify if the time interval rule/constraint is fulfilled. In case the
across_split option is selected (time interval rule is also checked across partitions)
"""
def create_datasets(path, logger, log_dir, train_split: float, across_split=True, interval=5):
	#all sub-folder names are automatically the labels for every image sample below
	labels = os.listdir(path)

	logger.info("Class Labels: " + str(labels))

	interval = interval
	unused_all = []
	train_all, valid_all, test_all = [], [], []

	ds = open(f"{log_dir}/DATASET.log", "w")

	for label in labels:
		unused = []
		all_filepaths = []

		#load all images per sub-folder and store label plus filepath within a list, which is the baseline for partitioning
		for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, label)):
			for fn in sorted(filenames):
				if not fn.lower().endswith(".jpg"):
					continue
				fp = os.path.abspath(os.path.join(dirpath, fn))
				all_filepaths.append(fp)

		n = len(all_filepaths)
		n_train = math.floor(n * train_split)
		n_valid = math.floor((n - n_train) / 2)
		n_test = (n-(n_valid+n_train))

		ds.write(f"\n--- Label {label} ---\nOriginal Image Number: {n}\nTraining Samples: {n_train}\nValidation Samples: {n_valid}\nTest Samples: {n_test}\nSum Data Split: {n_train+n_test+n_valid}\n")

		if not across_split:
			random.shuffle(all_filepaths)
			random.shuffle(all_filepaths)
			random.shuffle(all_filepaths)

		train_fp, valid_fp, test_fp = all_filepaths[:n_train], all_filepaths[n_train: n_train + n_valid], all_filepaths[n_train+n_valid:]

		logger.info("\n\n--- Label " + str(label) + "---\n")
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

		#apply interval rule to the validation and test set
		if valid_fp:
			for fp in valid_fp[:]:
				if fp_old is not None and date_and_artist_are_equal(fp, fp_old):
					fn_old = fp_old.replace("/", "\\").split("\\")[-1]
					fn = fp.replace("/", "\\").split("\\")[-1]

					if not check_time_interval(img1_fn=fn, img2_fn=fn_old, interval=interval):
						unused.append(fp)
						valid_fp.remove(fp)
				fp_old = fp
		else:
			pass

		fp_old = None

		if test_fp:
			for fp in test_fp[:]:
				if fp_old is not None and date_and_artist_are_equal(fp, fp_old):
					fn_old = fp_old.replace("/", "\\").split("\\")[-1]
					fn = fp.replace("/", "\\").split("\\")[-1]

					if not check_time_interval(img1_fn=fn, img2_fn=fn_old, interval=interval):
						unused.append(fp)
						test_fp.remove(fp)
				fp_old = fp
		else:
			pass

		#check if time interval rule is also valid across data splits, otherwise perform additional sorting of the affected partitions
		if across_split:

			if test_fp and valid_fp:
				if date_and_artist_are_equal(test_fp[0], valid_fp[-1]):
					test_fn = test_fp[0].replace("/", "\\").split("\\")[-1]
					val_fn = valid_fp[-1].replace("/", "\\").split("\\")[-1]

					if not check_time_interval(img1_fn=test_fn, img2_fn=val_fn, interval=interval):
						valid_fp, test_fp = sort_new_by_interval(valid_fp, test_fp, interval=interval)

			if train_fp and valid_fp:
				if date_and_artist_are_equal(valid_fp[0], train_fp[-1]):
					val_fn = valid_fp[0].replace("/", "\\").split("\\")[-1]
					train_fn = train_fp[-1].replace("/", "\\").split("\\")[-1]

					if not check_time_interval(img1_fn=val_fn, img2_fn=train_fn, interval=interval):
						train_fp, valid_fp = sort_new_by_interval(train_fp, valid_fp, interval=interval)

			if len(valid_fp) > len(train_fp):
				tmp = train_fp
				train_fp = valid_fp
				valid_fp = tmp

			if len(test_fp) > len(valid_fp):
				tmp = valid_fp
				valid_fp = test_fp
				test_fp = tmp

		n_train = len(train_fp)
		n_valid = len(valid_fp)
		n_test = len(test_fp)

		if len(train_fp) == 0 or len(valid_fp) == 0 or len(test_fp) == 0:
			logger.info("Number Training Files: " + str(n_train))
			logger.info("Number Validation Files: " + str(n_valid))
			logger.info("Number Testing Files: " + str(n_test))
			raise ValueError("At least one of the datasets is empty. Please, either add more files to the single distributions or check the time interval rule whether too much is be removed due to that rule. Abort...")

		logger.info("\n\n--- Label " + str(label) + "---\n")
		logger.info("Files after applying the interval rule, except for the training dataset!")
		logger.info("Original Image Number: " + str(n))
		logger.info("Number of Training Samples: " + str(n_train))
		logger.info("Training Samples: " + str(train_fp))
		logger.info("Number of Validation Samples: " + str(n_valid))
		logger.info("Validation Samples: " + str(valid_fp))
		logger.info("Number of Test Samples: " + str(n_test))
		logger.info("Test Samples: " + str(test_fp))
		logger.info("Sum Data Split: " + str(n_train + n_test + n_valid))

		train = [(label, fp) for fp in train_fp]
		valid = [(label, fp) for fp in valid_fp]
		test = [(label, fp) for fp in test_fp]

		random.shuffle(train)
		random.shuffle(valid)
		random.shuffle(test)

		ds.write(f"\n--- New number of samples after applying interval rule for {label} ---\nOriginal Image Number: {n}\nTraining Samples: {len(train)}\nValidation Samples: {len(valid)}\n"
				 f"Test Samples: {len(test)}\nSum Data Split: {len(train)+len(valid)+len(test)}\n")

		for x in train:
			train_all.append(x)
		for x in valid:
			valid_all.append(x)
		for x in test:
			test_all.append(x)
		for u in unused:
			unused_all.append(u)

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

	ds.write(f"\n\n--- Main sets including all classes ---\nOriginal Image Number: {train_all_n + valid_all_n + test_all_n}\nTraining Samples: {train_all_n}\n"
			 f"Validation Samples: {valid_all_n}\nTest Samples: {test_all_n}\nSum Data Split: {train_all_n + valid_all_n + test_all_n}\n")

	logger.info("Train-Set: " + str(train_all))
	logger.info("Val-Set: " + str(valid_all))
	logger.info("Test-Set: " + str(test_all))

	ds.write(f"\n\nTrain-Set: {train_all}\n\nVal-Set: {valid_all}\n\nTest-Set: {test_all}\n")

	ds.close()

	logger.info("\n\n------------ ALL files which were ignored by the interval rule ---------------")
	logger.info("Unused files due to the given time interval of " + str(interval) + " seconds.")
	logger.info("Number of unused files (validation and test set): " + str(len(unused_all)))
	logger.info("Unused files: " + str(unused_all))
	logger.info("---------------------------")

	return train_all, valid_all, test_all


"""
Creates the datasets (csv file), including the previously generated image files
"""
def generate_csv_files(data_split_dir: str, train: list, val: list, test: list):
	df_train = pd.DataFrame(data=train, columns=['Label', 'Filepath'])
	df_train.to_csv(os.path.join(data_split_dir, "train.csv"), header=True, index=None, sep=',')

	df_val = pd.DataFrame(data=val, columns=['Label', 'Filepath'])
	df_val.to_csv(os.path.join(data_split_dir, "val.csv"), header=True, index=None, sep=',')

	df_test = pd.DataFrame(data=test, columns=['Label', 'Filepath'])
	df_test.to_csv(os.path.join(data_split_dir, "test.csv"), header=True, index=None, sep=',')


"""
Creates the datasets (csv file), including the previously generated image files
"""
def load_from_csv(path, set='train'):
	if set == 'train':
		ds = pd.read_csv(filepath_or_buffer=os.path.join(path, 'train.csv'), sep=',')
	elif set == 'val':
		ds = pd.read_csv(filepath_or_buffer=os.path.join(path, 'val.csv'), sep=',')
	elif set == 'test':
		ds = pd.read_csv(filepath_or_buffer=os.path.join(path, 'test.csv'), sep=',')
	else:
		return None

	return ds.values

"""
Checks if all partition-specific csv files (train.csv, val.csv, test.csv) already exists or not
"""
def csv_datasets_exist(ds_dir):
	train_path = os.path.join(ds_dir, 'train.csv')
	val_path = os.path.join(ds_dir, 'val.csv')
	test_path = os.path.join(ds_dir, 'test.csv')

	return os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)

"""
Generates label dictionary based on the generated folders within the input image folder
"""
def get_label_dict(dirpath):
	labels = sorted(os.listdir(dirpath))
	labels_dict = {k: v for v, k in enumerate(labels)}

	return labels_dict

"""
Store and save label dictionary information to a given file in .json format
"""
def store_label_dict(ldict, storpath):
	ldict_json = json.dumps(ldict)
	f = open(storpath, "w")
	f.write(ldict_json)
	f.close()

"""
Load label dictionary information from .json format
"""
def load_label_dict(path):
	return json.load(open(path, 'r'))


"""
Load, augment, and pad (entire pre-processing) an input image to a desired image size with a square shape.
"""
def load_and_pad_img(img_path, desired_size=512, augmentation=False, max_augmentations=1, grayscale=False):
	im = Image.open(img_path, mode='r')
	w, h = im.size

	_to_pad = False
	if w != desired_size or h != desired_size:
		_to_pad = True

	if augmentation:
		im = augment(original=im, max_augmentations=max_augmentations)

	if _to_pad:
		old_size = im.size
		ratio = float(desired_size) / max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
		im = im.resize(new_size, Image.ANTIALIAS)

		if grayscale:
			new_im = Image.new("L", (desired_size, desired_size))
		else:
			new_im = Image.new("RGB", (desired_size, desired_size))
		new_im.paste(im, ((desired_size - new_size[0])//2, (desired_size-new_size[1])//2))

		return new_im

	else:
		if grayscale:
			im = im.convert("L")

		return im


"""
Load, augment, and pad (entire pre-processing) all images using the previously conducted data splits, together with the
ground truth annotation data
"""
class ImageListDataset(Dataset):

	def __init__(
			self,
			files_path,
			img_size=512,
			transform=None,
			ds='train',
			label_dict=None,
			augmentation=False,
			max_augmentations=1,
			grayscale=False,
	):
		if label_dict is None:
			raise ValueError("label_dict cannot be 'None'. Please pass a valid dictionary for mapping the labels.")

		if ds in ['train', 'val', 'test']:
			ds = load_from_csv(files_path, ds)
		else:
			raise ValueError(f"{ds} is no valid dataset name. Please choose either 'train', 'val' or 'test'.")

		self.img_filenames = ds[:, 1]
		self.label_files = ds[:, 0]
		self.label_dict = label_dict
		self.occurrences = collections.Counter(self.label_files)

		self.img_size = img_size
		self.transform = transform
		self.augmentation = augmentation
		self.max_augmentations = max_augmentations
		self.grayscale = grayscale

	def __getitem__(self, index):
		img_path = self.img_filenames[index].rstrip()

		resized_img = load_and_pad_img(
			img_path=img_path,
			desired_size=self.img_size,
			augmentation=self.augmentation,
			max_augmentations=self.max_augmentations,
			grayscale=self.grayscale,
		)
		if self.grayscale:
			resized_img.convert("L")

		img = transforms.ToTensor()(resized_img)

		label = self.label_files[index]

		if label not in self.label_dict.keys():
			label = 'Other'

		converted_label = self.label_dict[label]

		label = {"label": converted_label, "file_name": img_path}

		return img, label

	def __len__(self):
		if len(self.img_filenames) != len(self.label_files):
			raise ValueError("Number of images and labels is not the same. Abort...")
		return len(self.img_filenames)

	def get_label_string(self, idx):

		for k, v in self.label_dict.items():
			if idx == v:
				return k

		raise ValueError(f"Unknown label for idx {idx}.")


"""
Load, augment, and pad (entire pre-processing) all images from a given input image folder
"""
class ImageFolder(Dataset):
	def __init__(self, folder_path, img_size=512):
		if not os.path.exists(folder_path):
			raise ValueError("folder_path does not exist. Please set a valid path. Abort...")

		(dirpath, _, filenames) = next(os.walk(folder_path))

		self.files = []

		for fn in filenames:
			if fn.lower().endswith(".jpg"):
				self.files.append(os.path.join(dirpath, fn))

		self.img_size = img_size

	def __getitem__(self, index):
		img_path = self.files[index % len(self.files)]

		img = load_and_pad_img(img_path, self.img_size)
		img = transforms.ToTensor()(img)

		return img, img_path

	def __len__(self):
		return len(self.files)
