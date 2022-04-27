"""
Module: train.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

"""
Code from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/train.py
License: GNU General Public License v3.0
Access Data: 06.06.2020, Last Access Date: 25.01.2022
Changes: Modified by Christian Bergler, Alexander Gebhard (continuous since 06.06.2020)
"""

import sys
import copy
import time
import argparse

from models.detector import *

from data.dataset import *

from utils.utils import *
from utils.config import *
from utils.logging import *

from terminaltables import AsciiTable
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.utils.data import DataLoader


"""
Convert String to Boolean
"""
def str2bool(v):
	if v.lower() in ("yes", "true", "t", "y", "1"):
		return True
	elif v.lower() in ("no", "false", "f", "n", "0"):
		return False
	else:
		raise argparse.ArgumentTypeError("Boolean value expected.")


"""
Loading file paths of a given input file line-per-line 
"""
def load_data_files(file_path):
	with open(file_path, "r") as file:
		img_files = file.readlines()
	file.close()
	return img_files


"""
Evaluate detection model using the current state of a model together with the evaluation data and dataloader
"""
def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, img_size):
	model.eval()

	labels = []
	sample_metrics = []  # List of tuples (TP, confs, pred)
	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

		# Extract labels
		labels += targets[:, 1].tolist()
		# Rescale target
		targets[:, 2:] = xywh2xyxy(targets[:, 2:])
		targets[:, 2:] *= img_size

		imgs = Variable(imgs.type(Tensor), requires_grad=False)

		with torch.no_grad():
			outputs = model(imgs)
			outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

		sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

	if sample_metrics:
		true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
		precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
		return precision, recall, AP, f1, ap_class
	else:
		return np.array([0], dtype=np.float64), np.array([0], dtype=np.float64), np.array([0], dtype=np.float64), np.array([0], dtype=np.float64), np.array([1], dtype=np.int32)

"""
Save the trained model and corresponding options.
"""
def save_model(model, path):
	model = model.cpu()
	detector_state_dict = model.state_dict()

	save_dict = {
		"detectorState": detector_state_dict
	}
	if not os.path.isdir(ARGS.summary_dir):
		os.makedirs(ARGS.summary_dir)
	torch.save(save_dict, path)
	log.debug("Model successfully saved via torch save: " + str(path))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--epochs",
		type=int,
		default=1000,
		help="Number of Training Epoch"
	)

	parser.add_argument(
		"--batch_size",
		type=int,
		default=8,
		help="The number of images per batch"
	)

	parser.add_argument(
		"--gradient_accumulations",
		type=int,
		default=2,
		help="Number of gradient accumulations before optimization weight update step"
	)

	parser.add_argument(
		"--model_cfg",
		type=str,
		help="Path to YOLO model configuration file"
	)

	parser.add_argument(
		"--data_config",
		type=str,
		help="Path to YOLO data configuration file"
	)

	parser.add_argument(
		"--pretrained_weights",
		type=str,
		help="Continue Training based on pre-trained weights, either a previous model checkpoint (.pth file) or YOLO.weights (e.g. trained on ImageNet)"
	)

	parser.add_argument(
		"--n_cpu",
		type=int,
		default=1,
		help="Number of workers during data-loading and batch generation"
	)

	parser.add_argument(
		"--img_size",
		type=int,
		default=416,
		help="Network input image size (squared)"
	)

	parser.add_argument(
		"--checkpoint_interval",
		type=int,
		default=1,
		help="Number of epochs until a checkpoint is generated"
	)

	parser.add_argument(
		"--evaluation_interval",
		type=int,
		default=1,
		help="Number of epochs until the  evaluations on the validation set"
	)

	parser.add_argument(
		"--multiscale_training",
		default=False,
		help="Training with random generated image-sizes (based on the given img_size) after every tenth batch otherwise constant image size"
	)

	parser.add_argument(
		"--learning_rate",
		type=float,
		default=0.001,
		help="Initial network learning rate"
	)

	parser.add_argument(
		"--conf_thres",
		type=float,
		default=0.5,
		help="Objectness score threshold - describes the probability that an object is present inside a given bounding box"
	)

	parser.add_argument(
		"--log_dir",
		type=str,
		default=None,
		help="Path to directory where training logs are stored"
	)

	parser.add_argument(
		"--summary_dir",
		type=str,
		help="Path to directory where training summaries are stored",
	)

	parser.add_argument(
		"--debug",
		dest="debug",
		action="store_true",
		help="Log additional training and model information",
	)

	parser.add_argument(
		"--early_stopping_patience_epochs",
		metavar="N",
		type=int,
		default=7,
		help="Early stopping (stop training) after N epochs without any improvements on the validation set",
	)

	parser.add_argument(
		"--lr_patience_epochs",
		type=int,
		default=4,
		help="Decay the learning rate after N epochs without any improvements on the validation set",
	)

	parser.add_argument(
		"--lr_decay_factor",
		type=float,
		default=0.5,
		help="Decay factor applied to the learning rate",
	)

	parser.add_argument(
		"--augmentation",
		type=str2bool,
		default=True,
		help="Whether to augment the input data. Validation and test data will not be augmented",
	)

	parser.add_argument(
		"--photographer_and_date_constraint",
		type=str2bool,
		default=False,
		help="No images of the same photographer and date within the validation and/or test set (except training). By default this option is not used, because it might remove many of the existing data",
	)

	parser.add_argument(
		"--image_folder",
		type=str,
		default=None,
		help="Folder with the original image data to build a new data split. If this parameter is used the network will generate and save a new train.txt, valid.txt, and test.txt split within that folder as well as within the respective data config file. If this option is not used an existing data split has to be present and also within the data config file."
	)

	parser.add_argument(
		"--split",
		type=float,
		default=0.7,
		help="The split for the datasets. E.g. 0.7 => 0.7 for train, 0.15 for validation and 0.15 for test.",
	)

	ARGS = parser.parse_args()

	log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)
	summary_writer = SummaryWriter(ARGS.summary_dir)

	log.debug(vars(ARGS))

	epochs = ARGS.epochs
	batch_size = ARGS.batch_size
	grad_accum = ARGS.gradient_accumulations
	model_cfg = ARGS.model_cfg
	data_cfg = ARGS.data_config
	pretrained_weights = ARGS.pretrained_weights
	n_cpu = ARGS.n_cpu
	img_size = ARGS.img_size
	checkpoint_interval = ARGS.checkpoint_interval
	evaluation_interval = ARGS.evaluation_interval
	multiscale_training = ARGS.multiscale_training
	learning_rate = ARGS.learning_rate
	obj_conf_thres = ARGS.conf_thres
	log_dir = ARGS.log_dir
	summary_dir = ARGS.summary_dir
	debug = ARGS.debug
	lr_patience_epochs = ARGS.lr_patience_epochs
	patience_lr = math.ceil(lr_patience_epochs / evaluation_interval)
	patience_lr = int(max(1, patience_lr))
	lr_decay = ARGS.lr_decay_factor
	early_stopping = ARGS.early_stopping_patience_epochs
	augmentation = ARGS.augmentation
	image_folder = ARGS.image_folder
	train_split = ARGS.split
	photographer_and_date_constraint = ARGS.photographer_and_date_constraint

	log.info(f"Number of Training Epochs: {epochs}")
	log.info(f"Batch Size: {batch_size}")
	log.info(f"Number of Gradient Accumulations: {grad_accum}")
	log.info(f"YOLOv3 Model Config and Architecture: {model_cfg}")
	log.info(f"YOLOv3 Data Config and Architecture: {data_cfg}")
	log.info(f"YOLOv3 Type of Pre-Trained Weights ImgNet: {pretrained_weights}")
	log.info(f"Number of CPUs for Batch-Generation: {n_cpu}")
	log.info(f"Input Image Size: {img_size}")
	log.info(f"Number of Epochs for Checkpoint-Writing: {checkpoint_interval}")
	log.info(f"Number of Epochs for Validation: {evaluation_interval}")
	log.info(f"Multiscale Training: {multiscale_training}")
	log.info(f"Initial Learning Rate: {learning_rate}")
	log.info(f"Objective Confidence Threshold: {obj_conf_thres}")
	log.info(f"Directory of the Logging Output: {log_dir}")
	log.info(f"Learning Rate Patience Epochs: {patience_lr}")
	log.info(f"Learning Rate Decay: {lr_decay}")
	log.info(f"Early Criterion/Patience in Epochs: {early_stopping}")
	log.info(f"Directory of the Final Model and Additional Information Files: {summary_dir}")
	log.info(f"Percentage of Training Split (1-Training Split)/2 = Val/Test Split Fraction (only used together with --image_folder option): {train_split}")
	log.info(f"Photographer and date constraint within the validation and/or test set (except training): {photographer_and_date_constraint}")

	if image_folder is not None:
		log.info(f"Directory to the Original Image Data Folder to Compute/Store New Data Split: {image_folder}")
	else:
		log.info(f"Image Folder Option (--image_folder) is not set, therefore an Image Data Split Already Exist and is not Computed!")

	input_shape = (batch_size, 3, img_size, img_size)

	log.info(f"Network Input Shape: {input_shape}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	os.makedirs(summary_dir, exist_ok=True)

	os.makedirs(summary_dir + "/"f"checkpoints_{evaluation_interval}_conf{obj_conf_thres}", exist_ok=True)

	data_config = parse_data_config(data_cfg)

	if image_folder is not None:
		try:
			train, val, test = create_data_split(path=image_folder, logger=log, log_dir=log_dir, train_split=train_split, photographer_and_date_constraint=photographer_and_date_constraint)
			train_path, valid_path, test_path = generate_csv_files(data_split_dir=image_folder, train=train, val=val, test=test)
			data_config["train"] = train_path
			data_config["valid"] = valid_path
			data_config["test"] = test_path
			if write_to_data_config(data_cfg, data_config):
				log.info("Update process of file: " + data_cfg + " has been successfully processed")
			else:
				log.error("Update process of file: " + data_cfg + " has not been processed")
			log.info("Created the data split and the corresponding train.txt, valid.txt, and test.txt files at: " + str(image_folder))
		except Exception as exc:
			log.error(exc)
			log.close()
			sys.exit()

	train_path = data_config["train"]
	valid_path = data_config["valid"]
	test_path = data_config["test"]

	train_set = load_data_files(train_path)
	val_set = load_data_files(valid_path)
	test_set = load_data_files(test_path)

	class_names = load_classes(data_config["names"])

	prefix = "fin-detect"

	model = Detector(model_cfg).to(device)

	log.info("Setting up the Model")

	log.debug(model)

	model.apply(weights_init_normal)

	if pretrained_weights: #load weights from last checkpoint
		if pretrained_weights.endswith(".pth"):
			model.load_state_dict(torch.load(pretrained_weights))
		else: #load pre-trained weights for initial training
			model.load_pretrained_weights(pretrained_weights)

	data_dist = dict()
	data_dist["train"] = train_set
	data_dist["valid"] = val_set
	data_dist["test"] = test_set

	setnames = ['train', 'valid', 'test']

	datasets = {
		ds: Dataset(
			data_file_list=data_dist[ds],
			img_size=img_size,
			split=ds,
			augment=augmentation if ds == "train" else False,
			multiscale=multiscale_training
		)
		for ds in setnames
	}

	log.debug("Training Files: " + train_path)
	log.debug("Number of Files in Train: " + str(len(datasets["train"].img_files)))

	log.debug("Validation Files: " + valid_path)
	log.debug("Number of Files in Val: " + str(len(datasets["valid"].img_files)))

	log.debug("Test Files: " + test_path)
	log.debug("Number of Files in Test: " + str(len(datasets["test"].img_files)))

	dataloaders = {
		split: torch.utils.data.DataLoader(
			datasets[split],
			batch_size=1 if split == "test" else ARGS.batch_size,
			shuffle=False if split == "valid" or split == "test" else True,
			num_workers=1 if split == "valid" or split == "test" else n_cpu,
			drop_last=False if split == "valid" or split == "test" else True,
			pin_memory=True,
			collate_fn=datasets[split].collate_fn,
		)
		for split in setnames
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	best_model_state = None
	iou_train_val_test = 0.5
	nms_train_val_test = 0.5
	obj_conf_thres_test = 0.8
	obj_conf_thres_train_val = obj_conf_thres

	metrics = [
		"grid_size",
		"loss",
		"x",
		"y",
		"w",
		"h",
		"conf",
		"cls",
		"cls_acc",
		"recall50",
		"recall75",
		"precision",
		"conf_obj",
		"conf_noobj",
	]

	max_f1 = 0
	max_mAP = 0
	no_improvement_ctr = 0

	train_loader = dataloaders["train"]
	valid_loader = dataloaders["valid"]
	test_loader = dataloaders["test"]

	for epoch in range(epochs):
		model.train()
		start_time = time.time()

		for batch_i, (_, imgs, targets) in enumerate(train_loader):
			batches_done = len(train_loader) * epoch + batch_i

			imgs = Variable(imgs.to(device))

			targets = Variable(targets.to(device), requires_grad=False)

			loss, outputs = model(imgs, targets)

			loss.backward()

			if batches_done % grad_accum:
				optimizer.step()
				optimizer.zero_grad()

			log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, ARGS.epochs, batch_i, len(train_loader))

			metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

			# Log metrics at each YOLO layer
			for i, metric in enumerate(metrics):
				formats = {m: "%.6f" for m in metrics}
				formats["grid_size"] = "%2d"
				formats["cls_acc"] = "%.2f%%"
				row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
				metric_table += [[metric, *row_metrics]]

				tensorboard_log = []
				for j, yolo in enumerate(model.yolo_layers):
					for name, metric in yolo.metrics.items():
						if name != "grid_size":
							tensorboard_log += [(f"{name}_{j+1}", metric)]
				tensorboard_log += [("loss", loss.item())]

			log.write_scalar_summaries_logs(
				summary_writer=summary_writer,
				loss=loss.item(),
				metrics=tensorboard_log,
				lr=optimizer.param_groups[0]["lr"],
				epoch_time=time.time() - start_time,
				batch=batches_done,
				epoch=epoch,
				phase="train")

			log_str += AsciiTable(metric_table).table
			log_str += f"\nTotal loss: {loss.item()}"

			# Determine approximate time left for epoch
			epoch_batches_left = len(train_loader) - (batch_i + 1)
			time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
			log_str += f"\n---- Remaining Epoch Training Time: {time_left}"

			log.debug(log_str)

			model.seen += imgs.size(0)

		if epoch % evaluation_interval == 0 and epoch >= 1:

			log.debug("---- Evaluating Model on the Validation Dataset ----")

			log.debug(f"Epoch: {epoch}\n")

			precision, recall, AP, f1, ap_class = evaluate(
				model=model,
				dataloader=valid_loader,
				iou_thres=iou_train_val_test,
				conf_thres=obj_conf_thres_train_val,
				nms_thres=nms_train_val_test,
				img_size=img_size
			)

			evaluation_metrics = [
				("val_precision", precision.mean()),
				("val_recall", recall.mean()),
				("val_mAP", AP.mean()),
				("val_f1", f1.mean()),
			]

			log.write_scalar_summaries_logs(
				summary_writer=summary_writer,
				loss=None,
				metrics=evaluation_metrics,
				lr=optimizer.param_groups[0]["lr"],
				epoch_time=time.time() - start_time,
				batch=None,
				epoch=epoch,
				phase="valid")

			ap_table = [["Index", "Class name", "AP"]]
			for i, c in enumerate(ap_class):
				ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
			log.debug(AsciiTable(ap_table).table)

			log.debug("Evaluation Metrics:\n")
			log.debug(f"val_precision: {precision.mean()}")
			log.debug(f"val_recall: {recall.mean()}")
			log.debug(f"val_mAP: {AP.mean()}")
			log.debug(f"val_f1: {f1.mean()}")
			log.debug(f"---- mAP {AP.mean()}")

			f1_mean = f1.mean()
			AP_mean = AP.mean()

			if best_model_state is None:
				best_model_state = copy.deepcopy(model.state_dict())
				best_epoch = epoch

			if f1_mean > max_f1:
				no_improvement_ctr = 0
				max_f1 = f1_mean
				best_model_state = copy.deepcopy(model.state_dict())
				best_epoch = epoch
			else:
				no_improvement_ctr += 1

		if epoch % checkpoint_interval == 0:
			torch.save(model.state_dict(), summary_dir + "/"f"checkpoints_{evaluation_interval}_conf{obj_conf_thres_train_val}/model-ckpt_%d.pth" % epoch)

		if no_improvement_ctr == patience_lr:
			log.info(f"\nNo improvement for {patience_lr} epochs. Reduce learning rate by factor {lr_decay}...")
			learning_rate *= lr_decay
		if no_improvement_ctr >= early_stopping:
			log.info(f"\nNo improvement for {early_stopping} epochs. Stop training...")
			break

	log.debug("---- Evaluating Model on the Test Dataset ----")

	log.debug(f"Epoch 1:\n")

	model.load_state_dict(best_model_state)

	precision, recall, AP, f1, ap_class = evaluate(
		model=model,
		dataloader=test_loader,
		iou_thres=iou_train_val_test,
		conf_thres=obj_conf_thres_test,
		nms_thres=nms_train_val_test,
		img_size=img_size,
	)
	evaluation_metrics = [
		("val_precision", precision.mean()),
		("val_recall", recall.mean()),
		("val_mAP", AP.mean()),
		("val_f1", f1.mean()),
	]

	log.write_scalar_summaries_logs(
		summary_writer=summary_writer,
		loss=None,
		metrics=evaluation_metrics,
		lr=None,
		epoch_time=time.time() - start_time,
		batch=None,
		epoch=1,
		phase="test")

	ap_table = [["Index", "Class name", "AP"]]
	for i, c in enumerate(ap_class):
		ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
	log.debug(AsciiTable(ap_table).table)

	log.debug("Evaluation Metrics:\n")
	log.debug(f"test_precision: {precision.mean()}")
	log.debug(f"test_recall: {recall.mean()}")
	log.debug(f"test_mAP: {AP.mean()}")
	log.debug(f"test_f1: {f1.mean()}")
	log.debug(f"---- mAP {AP.mean()}")

	path = os.path.join(summary_dir, prefix + ".pk")

	save_model(model, path)

	log.close()
