#!/usr/bin/env python3
"""
Module: crop.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

import os
import re
import argparse
import numpy as np


from utils.logging import Logger
from utils.exif import get_exif_data

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Reads, converts, and parses YOLOv3 data config file including data split, class name, number of class information
"""
def parse_data_config(path):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

"""
Loads class labels at 'path'
"""
def load_classes(path):
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

"""
Image interpolation to a given width and height
"""
def interpolate_img(img_path, w, h):
	img = Image.open(img_path)
	interpolated = img.resize((w, h), Image.LANCZOS)
	return interpolated

"""
Loading bounding box information from the given bounding box file, considering option with only images containing just 
single labeled information and/or maximum amount of bounding boxes per image
"""
def load_bboxes(filepath, single, max_boxes=None):
	boxes = []
	with open(filepath, 'r') as f:
		lines = f.readlines()

		# if only one animal per image is allowed, check number of BBs
		if single and len(lines) > 1:
			raise ValueError('Too much bounding boxes. There should be only one box.')

		if max_boxes is not None and len(lines) > max_boxes:
			raise ValueError(f'Too much bounding boxes. There should be only {max_boxes} box(es).')

		for line in lines:
			box = line.replace("\n", '').split(' ')
			box = [float(value) for value in box]
			box[0] = int(box[0])
			boxes.append(box)

	return np.array(boxes)


"""
Extract and store each of the given bounding boxes from the original image to the respective output directory, 
while taking care of not having zero padding within the cropped results. Next to the cropped sub-images also 
the corresponding bounding box information is stored.
"""
def extract_and_store_fins(filepath, output_folder_path, boxes, timestamp):
	# Extract date and time
	time = 'None'
	if timestamp == 'None':
		date = f"{filepath.split('.')[0].split('_')[-4]}"
		if re.search('[a-zA-Z]', date):
			date = 'None'
	else:
		date, time = timestamp.split(' ')
		date = date.replace(':', '-').strip()
		time = time.replace(':', '-').strip()

	# Open the image which should be cropped
	img = Image.open(filepath)

	w, h = img.size

	desired_size = 512
	if w < desired_size or h < desired_size:
		img = interpolate_img(img_path=filepath, w=max(desired_size, w), h=max(desired_size, h))
		w, h = img.size

	for index, box in enumerate(boxes):
		_check_overrun = False

		# rescale bbox coordinates
		box_label = int(box[0])
		box_x = w * box[1]
		box_y = h * box[2]
		old_box_w = box_w = w * box[3]
		old_box_h = box_h = h * box[4]
		integral = box_w * box_h


		# check for too large bounding boxes and set the affected side to the corresponding image side
		if box_w > w:
			old_box_w, box_w = w, w
		if box_h > h:
			old_box_h, box_h = h, h


		# Turn smaller BBs into bigger squares to increase the BB size
		# More background, but no zero padding necessary
		if box_w < desired_size and box_h < desired_size:
			ignore = True
			if w > desired_size and h > desired_size:
				box_w, box_h = desired_size, desired_size
				old_box_w, old_box_h = desired_size, desired_size
				ignore = False
			if h == desired_size:
				box_y = desired_size / 2
				box_h = desired_size
				old_box_h = desired_size
				ignore = False
			if w == desired_size:
				box_x = desired_size / 2
				box_w = desired_size
				old_box_w = desired_size
				ignore = False
			if ignore:
				log.error(f"The {index}. detected bounding box for the animal is inappropriate for classification. Ignore...")
		else:
			_check_overrun = True

		# set to greater value
		if box_w >= box_h:
			box_h = box_w
		else:
			box_w = box_h

		# Setting the params for the cropped image
		x1 = left = box_x - (box_w / 2)
		y2 = top = box_y - (box_h / 2)
		x2 = right = box_x + (box_w / 2)
		y1 = bottom = box_y + (box_h / 2)

		if left < 0 or right > w or top < 0 or bottom > h:
			_check_overrun = True

		if _check_overrun:
			# Handle overrunning bounding boxes, in order to avoid black borders
			if right > w:
				overrun = right - w
				right = w
				left -= overrun

			if bottom > h:
				overrun = bottom - h
				bottom = h
				top -= overrun

			if top < 0:
				overrun = -top
				top = 0
				bottom += overrun

			if left < 0:
				overrun = -left
				left = 0
				right += overrun

		# Check if BB still crops over the edge, if yes, do everything again choosing the smaller side, though
		if top < 0 or bottom > h:
			if old_box_w >= old_box_h:
				box_w = old_box_h
				box_h = old_box_h
			else:
				box_h = old_box_w
				box_w = old_box_w

			# Setting the params for the cropped image
			x1 = left = box_x - (box_w / 2)
			y2 = top = box_y - (box_h / 2)
			x2 = right = box_x + (box_w / 2)
			y1 = bottom = box_y + (box_h / 2)


			if _check_overrun:
				# Handle overrunning bounding boxes, in order to avoid black borders
				if right > w:
					overrun = right - w
					right = w
					left -= overrun

				if bottom > h:
					overrun = bottom - h
					bottom = h
					top -= overrun

				if top < 0:
					overrun = -top
					top = 0
					bottom += overrun

				if left < 0:
					overrun = -left
					left = 0
					right += overrun

		im_cropped = img.crop((left, top, right, bottom))

		if box_w > desired_size or box_h > desired_size:

			im = im_cropped.resize((desired_size, desired_size), Image.ANTIALIAS)
			im_cropped = im

		fn = filepath.replace("/", "\\").split("\\")[-1]

		filename_cropped = f"{fn.split('.')[0]}_{label_dict[box_label]}_cropped_{index}_{date}_{time}.{fn.split('.')[1]}"
		filename_cropped_txt = f"{fn.split('.')[0]}_{label_dict[box_label]}_cropped_{index}_{date}_{time}.txt"

		filepath_cropped = os.path.join(output_folder_path, filename_cropped)
		filepath_cropped_txt = os.path.join(output_folder_path, filename_cropped_txt)

		im_cropped.save(filepath_cropped)

		with open(filepath_cropped_txt, "w") as f:
			f.write(f"{box[1]} {box[2]} {box[3]} {box[4]}\n")
			f.write(f"{left} {top} {right} {bottom}\n")
			f.write(f"{index}\n")

		log.info("Successfully Extracted and Saved Sub-Image Number " + str(index+1) + ": " + str(filepath_cropped))
		log.info("Successfully Extracted and Saved Sub-Image BBox Info: " + str(index+1) + ": " + str(filepath_cropped_txt))


"""
Extract time information based on the image exif data information if available
"""
def get_timestamp(imgpath):
	exif_data = get_exif_data(imgpath)
	return exif_data[-1]

"""
Processing of the entire extraction procedure taking the original images, w.r.t. to each animal/individual, the bounding
box information and final output folder
"""
def crop_all_images(orig_img_path, bb_box_path, output_path, animals, single):

	for label in animals:
		for (dirpath, dirnames, filenames) in os.walk(os.path.join(orig_img_path, label)):
			for filename in filenames:

				if not filename.lower().endswith('.jpg'):
					continue

				bb_filename = filename.replace('.JPG', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')

				log.info(f"Image File, Bounding Box File: {filename} {bb_filename}")

				bb_filepath = os.path.join(os.path.join(bb_box_path, label), bb_filename)
				img_filepath = os.path.join(os.path.join(orig_img_path, label), filename)
				try:
					boxes = load_bboxes(bb_filepath, single)
				except Exception as e:
					log.error(e)
					continue

				timestamp = get_timestamp(img_filepath)

				output_folder_path = os.path.join(output_path, label)
				if not os.path.exists(output_folder_path):
					os.makedirs(output_folder_path, exist_ok=True)

				extract_and_store_fins(filepath=img_filepath, output_folder_path=output_folder_path, boxes=boxes, timestamp=timestamp)

				log.info("Extraction Process of FIN-EXTRACT successfully finished!")

				log.close()


if __name__=='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"-d",
		"--debug",
		dest="debug",
		action="store_true",
		help="Log additional training and model information.",
	)

	parser.add_argument(
		"--single",
		dest="single",
		action="store_true",
		help="Verification of single labels and only one bounding box allowed",
	)

	parser.add_argument(
		"--log_dir",
		type=str,
		default=None,
		help="Path to directory where training logs are stored"
	)

	parser.add_argument(
		"--original_image_path",
		type=str,
		help="The path to the folder which contains the original images"
	)

	parser.add_argument(
		"--bounding_box_path",
		type=str,
		help="The path to the folder which contains the corresponding bounding boxes (can be also equal to --original_image_path, in case bbox-files are located there)"
	)

	parser.add_argument(
		"--data_config",
		type=str,
		help="Path to YOLO data configuration file"
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		help="Path to directory where the output files are stored",
	)

	ARGS = parser.parse_args()

	single = ARGS.single
	debug = ARGS.debug
	log_dir = ARGS.log_dir
	original_image_path = ARGS.original_image_path
	bounding_box_path = ARGS.bounding_box_path
	data_config = ARGS.data_config
	output_dir = ARGS.output_dir

	log = Logger("FIN-EXTRACT.log", debug, log_dir)

	log.info(f"Logging Mode: {debug}")
	log.info(f"Single Label Verification (Labels == Number Bounding Box: {single}")
	log.info(f"Logging Directory: {log_dir}")
	log.info(f"Original Input Image Directory: {original_image_path}")
	log.info(f"Bounding Box Directory: {bounding_box_path}")
	log.info(f"YOLOv3 Data Config and Architecture: {data_config}")
	log.info(f"Output Directory: {output_dir}")

	data_config = parse_data_config(data_config)
	class_names = load_classes(data_config["names"])

	label_dict = dict()
	for index in range(len(class_names)):
		label_dict[index] = class_names[index]

	os.makedirs(original_image_path, exist_ok=True)
	os.makedirs(bounding_box_path, exist_ok=True)

	animals = os.listdir(original_image_path)

	log.info("Start Extraction!\n--------------------")
	crop_all_images(orig_img_path=original_image_path, bb_box_path=bounding_box_path, output_path=output_dir, animals=animals, single=single)
