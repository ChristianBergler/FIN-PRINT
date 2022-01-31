#!/usr/bin/env python3
"""
Module: predict.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.logging import Logger
from collections import OrderedDict
from models.classifier import DefaultClassifierOpts, Classifier
from models.residual_encoder import DefaultEncoderOpts, ResidualEncoder as Encoder

from data.image_dataset import (
	ImageFolder,
	load_label_dict,
	DefaultImgDatasetOps
)

parser = argparse.ArgumentParser()

parser.add_argument(
	"--debug",
	dest="debug",
	action="store_true",
	help="Print additional training and model information.",
)

parser.add_argument(
	"--model_path",
	type=str,
	default=None,
	help="Path to a trained model.",
)

parser.add_argument(
	"--checkpoint_path",
	type=str,
	default=None,
	help="Path to a checkpoint to load the weights, which will be used instead of the model.",
)

parser.add_argument(
	"--output_path",
	type=str,
	default=None,
	help="The path (i.e. folder) where the predictions shall be stored."
)

parser.add_argument(
	"--labels_info_file",
	type=str,
	default=None,
	help="The path of the stored label dictionary."
)

parser.add_argument(
	"--log_dir",
	type=str,
	default=None,
	help="The directory to store the logs."
)

parser.add_argument(
	"--image_input_folder",
	type=str,
	default=None,
	help="Directory containing the images which should be classified."
)

parser.add_argument(
	"--threshold",
	type=float,
	default=None,
	help="Threshold for the probability of classifying a target class.",
)

parser.add_argument(
	"--batch_size",
	type=int,
	default=1,
	help="The number of images per batch."
)

parser.add_argument(
	"--img_size",
	type=int,
	default=512,
	help="Desired size of the squared image."
)

parser.add_argument(
	"--topK",
	type=int,
	default=1,
	help="Calculate the topK best predictions/probabilities."
)

parser.add_argument(
	"--num_workers",
	type=int,
	default=0,
	help="Number of workers used in data-loading"
)

parser.add_argument(
	"--no_cuda",
	dest="cuda",
	action="store_false",
	help="Do not use cuda to train model.",
)

ARGS = parser.parse_args()

log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)

models = {"encoder": 1, "classifier": 2}

"""
Convert prediction into label name while using the label dictionary information
"""
def get_label_string(labels_dict, pred):
	for k, v in labels_dict.items():
		if pred == v:
			return k

	raise ValueError(f"Unknown label for idx {pred}.")


"""
Main function to compute the (multi-class) predictions for each file, including data loading/pre-processing, model loading/initialization, prediction 
and final evaluation.
"""
if __name__ == "__main__":

	model_path = ARGS.model_path
	checkpoint_path = ARGS.checkpoint_path
	output_path = ARGS.output_path
	labels_info_file = ARGS.labels_info_file
	log_dir = ARGS.log_dir
	image_input_folder = ARGS.image_input_folder
	threshold = ARGS.threshold
	batch_size = ARGS.batch_size
	img_size = ARGS.img_size
	k = ARGS.topK
	num_workers = ARGS.num_workers

	log.info(f"Model Path: {model_path}")
	log.info(f"Checkpoint Path: {checkpoint_path}")
	log.info(f"Output Path: {output_path}")
	log.info(f"Path to the Label Info File: {labels_info_file}")
	log.info(f"Directory of the Logging Information: {log_dir}")
	log.info(f"Directory of the Prediction Input Images: {image_input_folder}")
	log.info(f"Threshold for Valid Prediction (None=ArgMax of Probability Vector): {threshold}")
	log.info(f"Batch Size: {batch_size}")
	log.info(f"Image Size: {img_size}")
	log.info(f"Amount of Network Hypothesis (top-K): {k}")

	log.debug("Setting up the Model")

	encoderOpts = DefaultEncoderOpts
	encoderOpts["input_channels"] = 3

	classifierOpts = DefaultClassifierOpts

	os.makedirs(output_path, exist_ok=True)

	labels_dict = load_label_dict(labels_info_file)
	inverted_labels_dict = {v: k for k, v in labels_dict.items()}

	log.info(f"Label dictionary: {labels_dict}")
	log.info(f"Inverted_labels_dict: {inverted_labels_dict}")

	if ARGS.checkpoint_path is not None:
		log.debug("Restoring checkpoint from {} instead of using a model file.".format(ARGS.checkpoint_path))
		checkpoint = torch.load(ARGS.checkpoint_path, map_location="cpu")
		encoder = Encoder(encoderOpts)
		encoder_out_ch = img_size * encoder.block_type.expansion
		num_classes = classifierOpts["num_classes"]
		classifierOpts["input_channels"] = encoder_out_ch
		classifier = Classifier(classifierOpts)
		log.debug(f"Classifier Opts: {classifierOpts}")

		model = nn.Sequential(OrderedDict([("encoder", encoder), ("classifier", classifier)]))
		model.load_state_dict(checkpoint["modelState"])
		log.warning("Using default preprocessing options. Provide Model file if they are changed")
		dataOpts = DefaultImgDatasetOps
	else:
		log.debug(f"Use model path {ARGS.model_path}")
		model_dict = torch.load(ARGS.model_path)
		encoder = Encoder(model_dict["encoderOpts"])
		encoder.load_state_dict(model_dict["encoderState"])
		classifier = Classifier(model_dict["classifierOpts"])
		classifier.load_state_dict(model_dict["classifierState"])
		num_classes = model_dict["num_classes"]
		model = nn.Sequential(OrderedDict([("encoder", encoder), ("classifier", classifier)]))
		dataOpts = model_dict["dataOpts"]

	log.info(model)

	if num_classes == 2:
		prefix = "vvi-detect"
		k = 1
	elif num_classes > 2:
		prefix = "fin-identify"
		if k > num_classes:
			k = num_classes
	else:
		raise Exception("Invalid Number of Classes!")

	log.info(f"Network Mode: {prefix}")

	if torch.cuda.is_available() and ARGS.cuda:
		model = model.cuda()
	model.eval()

	log.info("Predicting files from folder path {} ".format(image_input_folder))

	dataset = ImageFolder(
		folder_path=image_input_folder,
		img_size=img_size
	)

	data_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=ARGS.batch_size,
		num_workers=ARGS.num_workers,
		pin_memory=True,
	)

	log.info("Image size = {}".format(img_size))

	with torch.no_grad():
		for i, (input, img_path) in enumerate(data_loader):

			if torch.cuda.is_available() and ARGS.cuda:
				input = input.cuda()

			out = model(input).cpu()

			for n in range(out.shape[0]):
				start = time.time()

				file_path = img_path[n]

				log.info(f"Image path: {file_path}")

				probs = torch.nn.functional.softmax(out, dim=1).numpy()[n]

				argmax = np.argmax(probs, axis=0)

				pred = argmax
				max_prob = probs[argmax]
				animal = inverted_labels_dict[pred]

				if threshold is not None:
					thres_pass = max_prob >= ARGS.threshold
					if thres_pass:
						thres_pass = "Yes"
					else:
						thres_pass = "No"
				else:
					thres_pass = "No Threshold Set"

				fn = file_path.replace("/", "\\").split("\\")[-1]

				if num_classes > 2:
					top_k = torch.topk(out.data, k=k, dim=1, largest=True, sorted=True)
					top_k_values, top_k_indices = top_k
					top_k_values = top_k_values.numpy()[0]
					top_k_indices = top_k_indices.numpy()[0]

					log.info("Filename: {} | Prediction: {} | Probability={:.5f} | Threshold Passed: {}".format(fn, animal, max_prob, thres_pass))

					animals_top = [inverted_labels_dict[idx] for idx in top_k_indices]

					animals_probs = [probs[idx] for idx in top_k_indices]

					for idx in range(k):
						log.info("{}. Place: {} | Probability: {:.5f}".format(idx+1, animals_top[idx], animals_probs[idx]))
					log.info("")

					txt_fn = file_path.replace("/", "\\").split("\\")[-1]
					txt_fn = txt_fn.split(".")[0] + "_prediction.txt"
					prediction_file_fn = txt_fn

					prediction_file_dir = output_path
					os.makedirs(prediction_file_dir, exist_ok=True)
					prediction_file_path = os.path.join(prediction_file_dir, prediction_file_fn)

					with open(prediction_file_path, "w") as f:
						for key, prob in zip(animals_top, animals_probs):
							f.write("Filename: {} | Prediction: {} | Probability={:.5f} | Threshold Passed: {}".format(fn, key, prob, thres_pass)+"\n")
				else:
					log.info("Filename: {} | Prediction: {} | Probability={:.5f} | Threshold Passed: {}".format(fn, animal, max_prob, thres_pass))

					txt_fn = file_path.replace("/", "\\").split("\\")[-1]
					txt_fn = txt_fn.split(".")[0] + "_prediction.txt"
					prediction_file_fn = txt_fn

					prediction_file_dir = output_path
					os.makedirs(prediction_file_dir, exist_ok=True)
					prediction_file_path = os.path.join(prediction_file_dir, prediction_file_fn)

					with open(prediction_file_path, "w") as f:
						f.write("Filename: {} | Prediction: {} | Probability={:.5f} | Threshold Passed: {}".format(fn, animal, max_prob, thres_pass)+"\n")

				end = time.time()
				log.debug("Prediction Time: {}".format(end - start))

		log.debug("Finished proccessing")

	log.close()
