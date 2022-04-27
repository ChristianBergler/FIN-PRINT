#!/usr/bin/env python3
"""
Module: main.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import sys
import math
import argparse
import utils.metrics as m

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict
from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.classifier import Classifier, DefaultClassifierOpts

parser = argparse.ArgumentParser()

from data.image_dataset import (
	get_label_dict,
	create_datasets,
	ImageListDataset,
	store_label_dict,
	generate_csv_files,
	csv_datasets_exist,
	DefaultImgDatasetOps,
)

"""
Convert string to boolean.
"""
def str2bool(v):
	if v.lower() in ("yes", "true", "t", "y", "1"):
		return True
	elif v.lower() in ("no", "false", "f", "n", "0"):
		return False
	else:
		raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument(
	"--debug",
	dest="debug",
	action="store_true",
	help="Log additional training and model information.",
)

""" Directory parameters """
parser.add_argument(
	"--data_split_dir",
	type=str,
	help="The path to the directory including the data split csv files.",
)

parser.add_argument(
	"--data_dir",
	type=str,
	help="The path to the directory containing the image data.",
)

parser.add_argument(
	"--cache_dir",
	type=str,
	help="The path to the dataset directory.",
)

parser.add_argument(
	"--model_dir",
	type=str,
	help="The directory where the model will be stored.",
)

parser.add_argument(
	"--checkpoint_dir",
	type=str,
	help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
	"--log_dir",
	type=str,
	default=None,
	help="The directory to store the logs."
)

parser.add_argument(
	"--summary_dir",
	type=str,
	help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
	"--start_from_scratch",
	dest="start_scratch",
	action="store_true",
	help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
	"--max_train_epochs", type=int, default=500, help="The number of epochs to train for the classifier."
)

parser.add_argument(
	"--epochs_per_eval",
	type=int,
	default=2,
	help="The number of batches to run in between evaluations.",
)

parser.add_argument(
	"--batch_size",
	type=int,
	default=1,
	help="The number of images per batch."
)

parser.add_argument(
	"--num_workers",
	type=int,
	default=4,
	help="Number of workers used in data-loading"
)

parser.add_argument(
	"--no_cuda",
	dest="cuda",
	action="store_false",
	help="Do not use cuda to train model.",
)

parser.add_argument(
	"--lr",
	"--learning_rate",
	type=float,
	default=1e-5,
	help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
	"--beta1",
	type=float,
	default=0.5,
	help="beta1 for the adam optimizer."
)

parser.add_argument(
	"--lr_patience_epochs",
	type=int,
	default=8,
	help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
	"--lr_decay_factor",
	type=float,
	default=0.5,
	help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
	"--split",
	type=float,
	default=0.7,
	help="The split for the datasets. E.g. 0.7 => 0.7 for train, 0.15 for validation and 0.15 for test.",
)

parser.add_argument(
	"--early_stopping_patience_epochs",
	metavar="N",
	type=int,
	default=20,
	help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
	"--img_size",
	type=int,
	default=512,
	help="Desired size of the square image.")

parser.add_argument(
	"--num_classes",
	type=int,
	default=2,
	help="Number of classes to be trained and classified."
)

parser.add_argument(
	"--augmentation",
	type=str2bool,
	default=True,
	help="Whether to augment the input data. "
	"Validation and test data will not be augmented.",
)

parser.add_argument(
	"--grayscale",
	type=str2bool,
	default=False,
	help="Whether to process grayscale images or not."
)

parser.add_argument(
	"--create_datasets",
	type=str2bool,
	default=False,
	help="Whether to create new dataset splits for train, val, test or not.",
)

parser.add_argument(
	"--across_split",
	type=str2bool,
	default=True,
	help="Determines if the time interval rule is also checked across partitions (due to the splits) to ensure that images from the same photographer/date, fulfilling the interval constraint, are not spread across the partitions.",
)

parser.add_argument(
	"--interval",
	type=int,
	default=15,
	help="The allowed time interval between two photos."
)

parser.add_argument(
	"--model_path_ae",
	type=str,
	default=None,
	help="The path to a pretrained autoencoder.",
)

parser.add_argument(
	"--max_aug",
	type=int,
	default=1,
	help="Determines the maximum number of augmentations for one image. The final number is selected randomly."
)

parser.add_argument(
	"--resnet",
	dest="resnet_size",
	type=int,
	default=18,
	help="ResNet size"
)

parser.add_argument(
	"--conv_kernel_size",
	nargs="*",
	type=int,
	help="Initial convolution kernel size."
)

parser.add_argument(
	"--max_pool",
	type=int,
	default=None,
	help="Use max pooling after the initial convolution layer.",
)

parser.add_argument(
	"--topK",
	type=int,
	default=3,
	help="Top-K classification hypotheses considering for calculation of model accuracy (only in case of multi-class classification, num_classes > 2).",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

if ARGS.conv_kernel_size is not None and len(ARGS.conv_kernel_size):
	ARGS.conv_kernel_size = ARGS.conv_kernel_size[0]

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Save the trained model and corresponding options.
"""
def save_model(encoder, encoderOpts, classifier, classifierOpts, dataOpts, path, num_classes):
	encoder = encoder.cpu()
	classifier = classifier.cpu()
	encoder_state_dict = encoder.state_dict()
	classifier_state_dict = classifier.state_dict()

	save_dict = {
		"encoderOpts": encoderOpts,
		"classifierOpts": classifierOpts,
		"dataOpts": dataOpts,
		"encoderState": encoder_state_dict,
		"classifierState": classifier_state_dict,
		"num_classes": num_classes
	}
	if not os.path.isdir(ARGS.model_dir):
		os.makedirs(ARGS.model_dir)
	torch.save(save_dict, path)
	log.debug("Model successfully saved via torch save: " + str(path))

"""
Main function to compute data preprocessing, network training, evaluation, and final model saving.
"""
if __name__ == "__main__":

	log.debug(vars(ARGS))

	encoderOpts = DefaultEncoderOpts
	classifierOpts = DefaultClassifierOpts
	dataOpts = DefaultImgDatasetOps
	dataOpts = None

	for arg, value in vars(ARGS).items():
		if arg in encoderOpts and value is not None:
			encoderOpts[arg] = value
		if arg in classifierOpts and value is not None:
			classifierOpts[arg] = value

	if ARGS.grayscale:
		encoderOpts["input_channels"] = 1
	else:
		encoderOpts["input_channels"] = 3

	create_csv_files = ARGS.create_datasets
	across_split = ARGS.across_split
	interval = ARGS.interval
	train_split = ARGS.split
	batch_size = ARGS.batch_size
	num_classes = ARGS.num_classes
	img_size = ARGS.img_size
	data_split_dir = ARGS.data_split_dir
	data_dir = ARGS.data_dir
	max_augmentations = ARGS.max_aug
	grayscale = ARGS.grayscale
	summary_dir = ARGS.summary_dir
	model_dir = ARGS.model_dir
	log_dir = ARGS.log_dir
	topK = ARGS.topK

	if num_classes == 2:
		prefix = "vvi-detect"
		topK = 1
	elif num_classes > 2:
		prefix = "fin-identify"
		if topK > num_classes:
			topK = num_classes
	else:
		raise Exception("Invalid Number of Classes!")

	if not os.path.exists(data_split_dir):
		os.mkdir(data_split_dir)
	if not os.path.exists(data_dir):
		os.mkdir(data_dir)

	ARGS.lr *= ARGS.batch_size
	patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)
	patience_lr = int(max(1, patience_lr))
	classifierOpts["num_classes"] = num_classes

	log.info(f"ResNet Architecture: {ARGS.resnet_size}")
	log.info(f"Initial Convolutional-Kernel Size: {ARGS.conv_kernel_size}")
	log.info(f"Input Image Size: {img_size}")
	log.info(f"Input channels: {encoderOpts['input_channels']}")
	log.info(f"Create Data Split and New CSV Files: {create_csv_files}")
	log.info(f"Time Interval Between Images: {interval} sec")
	log.info(f"Time Interval Rule is also Checked Across Partitions/Splits: {across_split}")
	log.info(f"Percentage of Training Split (1-Training Split)/2 = Val/Test Split Fraction: {train_split}")
	log.info(f"Batch Size: {batch_size}")
	log.info(f"Initial Learning Rate: {ARGS.lr}")
	log.info(f"Early Criterion/Patience in Epochs: {ARGS.early_stopping_patience_epochs}")
	log.info(f"Number of Epochs for Validation: {ARGS.epochs_per_eval}")
	log.info(f"Number of Classes: {num_classes}")
	log.info(f"Augmentation: {ARGS.augmentation}")
	log.info(f"Number of Maximum Augmentations per Image: {max_augmentations}")
	log.info(f"Grayscale Input Images: {grayscale}")
	log.info(f"Directory of the Data Split Information: {data_split_dir}")
	log.info(f"Directory of the Input Image Data: {data_dir}")
	log.info(f"Directory of all the Summaries: {summary_dir}")
	log.info(f"Directory of the Logging Output: {log_dir}")
	log.info(f"Directory of the Final Model and Additional Information Files: {model_dir}")
	log.info(f"TopK Value for Accuracy Calculation (k-Classification Hypotheses): {topK}")

	input_shape = (batch_size, encoderOpts["input_channels"], img_size, img_size)

	log.info(f"Network Input Shape: {input_shape}")
	log.info("Setting up the Model")

	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)

	if ARGS.model_path_ae is not None:
		saved_model = torch.load(ARGS.model_path_ae)
		encoder = Encoder(saved_model["encoderOpts"])

		encoder2 = Encoder(saved_model["encoderOpts"])

		init_encoder_dict = encoder2.state_dict()

		for key in saved_model["encoderState"].keys():
			if key.startswith("layer4"):
				saved_model["encoderState"][key] = init_encoder_dict[key]

		log.info("Initializing classifier with pretrained weights.")
		encoder.load_state_dict(saved_model["encoderState"])
	else:
		encoder = Encoder(encoderOpts)
		encoder_dict = None

	log.debug("Encoder: " + str(encoder))

	encoder_out_ch = img_size * encoder.block_type.expansion

	log.debug("Encoder Out Channels: " + str(encoder_out_ch))

	classifierOpts["num_classes"] = num_classes

	classifierOpts["input_channels"] = encoder_out_ch

	classifier = Classifier(classifierOpts)

	log.debug("Classifier: " + str(classifier))

	if not csv_datasets_exist(data_split_dir) or create_csv_files:
		try:
			train, val, test = create_datasets(path=data_dir, logger=log, log_dir=log_dir, train_split=train_split, across_split=across_split, interval=interval)
			generate_csv_files(data_split_dir=data_split_dir, train=train, val=val, test=test)
			log.info("Created the CSV-Files...")
		except Exception as exc:
			log.error(exc)
			log.close()
			sys.exit()

	label_storpath = rf"label_dictionary_{num_classes}_interval{interval}.json"
	label_dict = get_label_dict(data_dir)
	store_label_dict(label_dict, model_dir+"/"+label_storpath)
	log.debug(f"Label dictionary: {label_dict}")
	
	setnames = ['train', 'val', 'test']

	datasets = {
		ds: ImageListDataset(
			files_path=data_split_dir,
			img_size=img_size,
			ds=ds,
			label_dict=label_dict,
			augmentation=ARGS.augmentation if ds == "train" else False,
			max_augmentations=max_augmentations,
			grayscale=grayscale,
		)
		for ds in setnames
	}

	log.debug("Number of Files in Train: " + str(len(datasets["train"].img_filenames)))
	log.debug("Label Distribution Train: " + str(datasets["train"].occurrences))

	log.debug("Number of Files in Val: " + str(len(datasets["val"].img_filenames)))
	log.debug("Label Distribution Val: " + str(datasets["val"].occurrences))

	log.debug("Number of Files in Test: " + str(len(datasets["test"].img_filenames)))
	log.debug("Label Distribution Test: " + str(datasets["test"].occurrences))

	dataloaders = {
		split: torch.utils.data.DataLoader(
			datasets[split],
			batch_size=1 if split == "test" else ARGS.batch_size,
			shuffle=True,
			num_workers=ARGS.num_workers,
			drop_last=False if split == "val" or split == "test" else True,
			pin_memory=True,
		)
		for split in setnames
	}

	model = nn.Sequential(
		OrderedDict([("encoder", encoder), ("classifier", classifier)])
	)

	trainer = Trainer(
		model=model,
		logger=log,
		prefix=prefix,
		checkpoint_dir=ARGS.checkpoint_dir,
		summary_dir=ARGS.summary_dir,
		n_summaries=4,
		start_scratch=ARGS.start_scratch,
		num_classes=num_classes,
		grayscale=grayscale,
		label_dict=label_dict,
		topK=topK
	)

	metrics = {
		"accuracy": m.Accuracy(ARGS.device),
	}

	optimizer = optim.Adam(
		model.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999)
	)

	metric_mode = "max"
	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode=metric_mode,
		patience=patience_lr,
		factor=ARGS.lr_decay_factor,
		threshold=1e-3,
		threshold_mode="abs",
	)

	model = trainer.fit(
		dataloaders["train"],
		dataloaders["val"],
		dataloaders["test"],
		loss_fn=nn.CrossEntropyLoss(),
		optimizer=optimizer,
		scheduler=lr_scheduler,
		n_epochs=ARGS.max_train_epochs,
		val_interval=ARGS.epochs_per_eval,
		patience_early_stopping=ARGS.early_stopping_patience_epochs,
		device=ARGS.device,
		metrics=metrics,
		val_metric="accuracy",
		val_metric_mode=metric_mode
	)

	encoder = model.encoder

	classifier = model.classifier

	path = os.path.join(ARGS.model_dir, prefix + ".pk")

	save_model(encoder, encoderOpts, classifier, classifierOpts, dataOpts, path, num_classes)

	log.close()
