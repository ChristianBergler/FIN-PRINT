"""
Module: predict.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

"""
Code from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/detect.py
License: GNU General Public License v3.0
Access Data: 06.06.2020, Last Access Date: 25.01.2022
Changes: Modified by Christian Bergler, Alexander Gebhard (continuous since 06.06.2020)
"""

import time
import datetime
import argparse

from utils.utils import *
from data.dataset import *
from utils.logging import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from models.detector import Detector


"""
Scaling of a given bounding using x1, y1, x2, y2 together with the original image size
into x, y, width, height with x and y being the centered coordinates of a given bounding box
"""
def scale_box(img_size, x1, x2, y1, y2):
	dw = 1. / img_size[0]
	dh = 1. / img_size[1]
	x = (x1 + x2) / 2.0
	y = (y1 + y2) / 2.0
	w = x2 - x1
	h = y2 - y1
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return round(x, 6), round(y, 6), round(w, 6), round(h, 6)

"""
Check if bounding box is out of scope/dimension
"""
def check_bbox_dimensions(bb_x1, bb_y1, bb_width, bb_height, img_width, img_height):
	if bb_x1 < 0:
		log.info("x coordinate of bounding box negative")
		return False
	elif bb_x1 > img_width:
		log.info("x coordinate of bounding box larger than original image width")
		return False
	elif bb_y1 < 0:
		log.info("y coordinate of bounding box negative")
		return False
	elif bb_y1 > img_height:
		log.info("y coordinate of bounding box larger than original image height")
		return False
	elif bb_width > img_width:
		log.info("width of bounding box larger than original image width")
		return False
	elif bb_height > img_height:
		log.info("height of bounding box larger than original image height")
		return False
	else:
		return True

"""
Use a trained model to predict bounding box(es) for a portion of given input images (image folder) and store detected image results plus bounding box information
"""
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--debug",
		dest="debug",
		action="store_true",
		help="Log additional training and model information")

	parser.add_argument(
		"--image_folder",
		type=str,
		help="path to dataset")

	parser.add_argument(
		"--model_cfg",
		type=str,
		help="Path to YOLO model configuration file")

	parser.add_argument(
		"--model_path",
		type=str,
		help="Path to the trained model (.pk file) or a given checkpoint (.pth file)")

	parser.add_argument(
		"--class_path",
		 type=str,
		 help="Path to the class label files (class.names)")

	parser.add_argument(
		"--conf_thres",
		type=float,
		default=0.8,
		help="Objectness score threshold - describes the probability that an object is present inside a given bounding box")

	parser.add_argument(
		"--nms_thres",
		type=float,
		default=0.5,
		help="Threshold for for non-maximum suppression algorithm")

	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="The number of images per batch")

	parser.add_argument(
		"--n_cpu",
		type=int,
		default=0,
		help="Number of workers during data-loading and batch generation")

	parser.add_argument(
		"--img_size",
		type=int,
		default=416,
		help="Network input image size (squared)")

	parser.add_argument(
		"--log_dir",
		type=str,
		default=None,
		help="Path to directory where training logs are stored")

	parser.add_argument("--output_dir", type=str, help="The folder to store all detection outputs.")

	ARGS = parser.parse_args()

	log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)

	debug = ARGS.debug
	image_folder = ARGS.image_folder
	model_cfg = ARGS.model_cfg
	model_path = ARGS.model_path
	class_path = ARGS.class_path
	n_cpu = ARGS.n_cpu
	obj_conf_thres = ARGS.conf_thres
	non_max_suppress = ARGS.nms_thres
	log_dir = ARGS.log_dir
	output_dir = ARGS.output_dir
	img_size = ARGS.img_size
	batch_size = ARGS.batch_size

	log.info(f"Logging Configuration: {debug}")
	log.info(f"Input Image Folder Path: {image_folder}")
	log.info(f"YOLOv3 Model Config and Architecture: {model_cfg}")
	log.info(f"Path to the Trained Model: {model_path}")
	log.info(f"Class Config File: {class_path}")
	log.info(f"Number of CPUs for Batch-Generation: {n_cpu}")
	log.info(f"Objective Confidence Threshold: {obj_conf_thres}")
	log.info(f"image_folder of the Logging Output: {log_dir}")
	log.info(f"Input Image Size: {img_size}")
	log.info(f"Batch Size: {batch_size}")
	log.info(f"Output Folder for final Model and Additional Information Files: {output_dir}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	start = time.time()

	wp = model_path.replace('/', '\\').split('\\')[-1]

	os.makedirs(output_dir, exist_ok=True)

	detector = Detector(model_cfg, img_size=img_size).to(device)

	log.info("Setting up the Model")

	if model_path.endswith(".pth"):
		model_dict = torch.load(model_path, map_location=device)
		detector.load_state_dict(model_dict)
	else:
		model_dict = torch.load(model_path, map_location=device)
		detector.load_state_dict(model_dict["detectorState"])

	log.debug(detector)

	detector.eval()

	image_loader = torch.utils.data.DataLoader(
		ImageFolder(image_folder, img_size=img_size),
		batch_size=batch_size,
		num_workers=n_cpu,
		shuffle=False,
	)

	classes = load_classes(class_path)  # Extracts class labels from file

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index

	log.info("Start performing object detection:\n")

	prev_time = time.time()

	for batch_i, (img_paths, input_imgs) in enumerate(image_loader):

		input_imgs = Variable(input_imgs.type(Tensor))

		try:
			with torch.no_grad():
				detections = detector(input_imgs)
				detections = non_max_suppression(detections, obj_conf_thres, non_max_suppress)
		except Exception as exc:
			log.error(f"Exception message for image="+img_paths+str(exc))
			continue

		current_time = time.time()
		inference_time = datetime.timedelta(seconds=current_time - prev_time)
		prev_time = current_time

		log.info("Detection File(s) in Batch: %s\nCurrent Batch: %d\nInference per Batch: %s\n" % (img_paths, batch_i, inference_time))

		imgs.extend(img_paths)
		img_detections.extend(detections)

	colors = [(1.0, 0.0, 0.0, 1.0)]

	log.info("\n")
	log.info("Saving Images together with Bounding Box Information:\n")

	for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

		log.info("(%d) Image: '%s'" % (img_i, path))

		img = np.array(Image.open(path))
		plt.figure()
		fig, ax = plt.subplots(1)
		ax.imshow(img)

		img_height = img.shape[0]
		img_width = img.shape[1]

		bbox_entries = []

		if detections is not None:
			detections = rescale_boxes(detections, img_size, img.shape[:2])
			unique_labels = detections[:, -1].cpu().unique()
			n_cls_preds = len(unique_labels)
			bbox_colors = random.sample(colors, n_cls_preds)

			for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in detections:

				log.info("Prediction Label: %s, Confidence: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

				box_w = x2 - x1
				box_h = y2 - y1

				if not check_bbox_dimensions(x1, y1, box_w, box_h, img_width, img_height):
					continue

				x, y, w, h = scale_box((img_width, img_height), float(x1), float(x2), float(y1), float(y2))
				bbox_info = "{} {} {} {} {}\n".format(int(cls_pred), x, y, w, h)
				bbox_entries.append(bbox_info)

				color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

				bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")

				ax.add_patch(bbox)

				plt.text(
					x1,
					y1,
					s=classes[int(cls_pred)],
					color="white",
					verticalalignment="top",
					bbox={"color": color, "pad": 0},
				)

		else:
			log.info("No Bounding Box detected!")

		plt.axis("off")
		plt.gca().xaxis.set_major_locator(NullLocator())
		plt.gca().yaxis.set_major_locator(NullLocator())

		filename = path.replace('\\', '/').split("/")[-1].split(".")[0]

		output_file_img = os.path.join(output_dir, f"{filename}.png")
		output_file_txt = os.path.join(output_dir, f"{filename}.txt")

		plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.0)

		with open(output_file_txt, 'w') as f:
			for entry in bbox_entries:
				f.writelines(entry)

		log.info("Image + BBox Info successfully saved to (%s, %s)" % (output_file_img, output_file_txt))
		log.info("------------------------------\n\n")

		plt.close()

	end = time.time()

	log.info(f"Final Processing Time: {end - start}")
