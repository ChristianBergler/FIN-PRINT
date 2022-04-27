"""
Module: trainer.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import copy
import math
import time
import operator
import platform
import numpy as np
import utils.metrics as m

import torch
import torch.nn as nn

from typing import Union
from utils.logging import Logger
from tensorboardX import SummaryWriter
from utils.confusionmeter import ConfusionMeter
from utils.checkpoints import CheckpointHandler
from utils.early_stopping import EarlyStoppingCriterion
from utils.summary import prepare_img, confusion_matrix_fig
from utils.metrics import Topk_accuracy, Topk_weighted_accuracy, calculate_metrics

"""
Class which implements network training, validation and testing as well as writing checkpoints, logs, summaries, and saving the final model.
"""
class Trainer:

	"""
	Initializing summary writer and checkpoint handler as well as setting required variables for training.
	"""
	def __init__(
		self,
		model: nn.Module,
		logger: Logger,
		label_dict,
		prefix: str = "",
		checkpoint_dir: Union[str, None] = None,
		summary_dir: Union[str, None] = None,
		n_summaries: int = 4,
		input_shape: tuple = None,
		start_scratch: bool = False,
		num_classes=12,
		grayscale=False,
		topK=3
	):
		self.topK = topK
		self.model = model
		self.logger = logger
		self.prefix = prefix

		self.num_classes=num_classes
		self.logger.info("Init summary writer")

		if summary_dir is not None:
			run_name = prefix + "_" if prefix != "" else ""
			run_name += "{time}-{host}".format(
				time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
				host=platform.uname()[1],
			)
			summary_dir = os.path.join(summary_dir, run_name)

		self.n_summaries = n_summaries
		self.writer = SummaryWriter(summary_dir)

		if input_shape is not None:
			dummy_input = torch.rand(input_shape)
			self.logger.info("Writing graph to summary")
			self.writer.add_graph(self.model, dummy_input)

		if checkpoint_dir is not None:
			self.cp = CheckpointHandler(
				checkpoint_dir, prefix=prefix, logger=self.logger
			)
		else:
			self.cp = None

		self.start_scratch = start_scratch

		self.grayscale = grayscale
		self.label_dict = label_dict
		self.label_dict_swapped = {self.label_dict[k]: k for k in self.label_dict}

	"""
	Starting network training from scratch or loading existing checkpoints. The model training and validation is processed for a given
	number of epochs while storing all relevant information (metrics, summaries, logs, checkpoints) after each epoch. After the training 
	is stopped (either no improvement of the chosen validation metric for a given number of epochs, or maximum training epoch is reached)
	the model will be tested on the independent test set and saved to the selected model target directory.
	"""
	def fit(
		self,
		train_loader,
		val_loader,
		test_loader,
		loss_fn,
		optimizer,
		scheduler,
		n_epochs,
		val_interval,
		patience_early_stopping,
		device,
		metrics: Union[list, dict] = [],
		val_metric: Union[int, str] = "loss",
		val_metric_mode: str = "min",
		start_epoch=0
	):

		self.logger.info("Init model on device '{}'".format(device))
		self.model = self.model.to(device)

		best_model = copy.deepcopy(self.model.state_dict())
		best_metric = 0.0 if val_metric_mode == "max" else float("inf")

		patience_stopping = math.ceil(patience_early_stopping / val_interval)
		patience_stopping = int(max(1, patience_stopping))
		early_stopping = EarlyStoppingCriterion(
			mode=val_metric_mode, patience=patience_stopping
		)

		if not self.start_scratch and self.cp is not None:
			checkpoint = self.cp.read_latest()
			if checkpoint is not None:
				try:
					try:
						self.model.load_state_dict(checkpoint["modelState"])
					except RuntimeError as e:
						self.logger.error(
							"Failed to restore checkpoint: "
							"Checkpoint has different parameters"
						)
						self.logger.error(e)
						raise SystemExit

					optimizer.load_state_dict(checkpoint["trainState"]["optState"])
					start_epoch = checkpoint["trainState"]["epoch"] + 1
					best_metric = checkpoint["trainState"]["best_metric"]
					best_model = checkpoint["trainState"]["best_model"]
					best_TWA = checkpoint["trainState"]["best_TWA"]
					best_TUA = checkpoint["trainState"]["best_TUA"]

					early_stopping.load_state_dict(
						checkpoint["trainState"]["earlyStopping"]
					)
					scheduler.load_state_dict(checkpoint["trainState"]["scheduler"])
					self.logger.info("Resuming with epoch {}".format(start_epoch))
				except KeyError:
					self.logger.error("Failed to restore checkpoint")
					raise

		since = time.time()

		self.logger.info("Start training model " + self.prefix)

		try:
			if val_metric_mode == "min":
				val_comp = operator.lt
			else:
				val_comp = operator.gt
			for epoch in range(start_epoch, n_epochs):
				self.train_epoch(
					epoch, train_loader, loss_fn, optimizer, metrics, device
				)
				if epoch % val_interval == 0 or epoch == n_epochs - 1:
					val_loss, _, val_TWA, val_TUA = self.test_epoch(
						epoch, val_loader, loss_fn, metrics, device, phase="val"
					)
					if val_metric == "loss":
						val_result = val_loss
					else:
						val_result = metrics[val_metric].get()
					if val_comp(val_result, best_metric):
						best_metric = val_result
						best_model = copy.deepcopy(self.model.state_dict())
						best_TWA = val_TWA
						best_TUA = val_TUA

					self.cp.write(
						{
							"modelState": self.model.state_dict(),
							"trainState": {
							"epoch": epoch,
							"best_metric": best_metric,
							"best_TWA": best_TWA,
							"best_TUA": best_TUA,
							"best_model": best_model,
							"optState": optimizer.state_dict(),
							"earlyStopping": early_stopping.state_dict(),
							"scheduler": scheduler.state_dict(),
							}
						}
					)
					scheduler.step(val_result)
					if early_stopping.step(val_result):
						self.logger.info(
							"No improvment over the last {} epochs. Stopping.".format(
								patience_early_stopping
							)
						)
						break
		except Exception:
			import traceback
			self.logger.warning(traceback.format_exc())
			self.logger.warning("Aborting...")
			self.logger.close()
			raise SystemExit

		self.model.load_state_dict(best_model)
		final_loss, final_ACC, final_TWA, final_TUA = self.test_epoch(
			0, test_loader, loss_fn, metrics, device, phase="test"
		)

		if val_metric == "loss":
			final_metric = final_loss
		else:
			final_metric = metrics[val_metric].get()

		time_elapsed = time.time() - since
		self.logger.info(
			"Training complete in {:.0f}m {:.0f}s".format(
				time_elapsed // 60, time_elapsed % 60
			)
		)

		if self.num_classes > 2:
			self.logger.info("Best Val Metric (Accuracy) {:4f}".format(best_metric))
			self.logger.info("Top-" + str(self.topK) + " Weighted Val Metric (Accuracy - TWA): {:4f}".format(best_TWA))
			self.logger.info("Top-" + str(self.topK) + " Unweighted Val Metric (Accuracy - TUA): {:4f}".format(best_TUA))
			self.logger.info("Best Test Metric (Accuracy) {:4f}".format(final_metric))
			self.logger.info("Top-" + str(self.topK) + " Weighted Test Metric (Accuracy - TWA): {:4f}".format(final_TWA))
			self.logger.info("Top-" + str(self.topK) + " Unweighted Test Metric (Accuracy - TUA): {:4f}".format(final_TUA))
		else:
			self.logger.info("Best val metric: {:4f}".format(best_metric))
			self.logger.info("Final test metric: {:4f}".format(final_metric))

		return self.model

	"""
	Training of one epoch using pre-extracted training data, loss function, optimizer, and respective metrics
	"""
	def train_epoch(self, epoch, train_loader, loss_fn, optimizer, metrics, device):
		self.logger.debug("train|{}|start".format(epoch))

		if isinstance(metrics, list):
			for metric in metrics:
				metric.reset(device)
		else:
			for metric in metrics.values():
				metric.reset(device)

		topk_acc = Topk_accuracy(device=device, k=self.topK)
		topk_acc.reset(device)

		topk_acc_rew = Topk_weighted_accuracy(device=device, k=self.topK)
		topk_acc_rew.reset(device)

		self.model.train()

		epoch_start = time.time()
		start_data_loading = epoch_start
		data_loading_time = m.Sum(torch.device("cpu"))
		epoch_loss = m.Mean(device)

		cm_train = ConfusionMeter(n_categories=self.num_classes)

		for i, (features, label) in enumerate(train_loader):

			call_label = label["label"].to(device, non_blocking=True, dtype=torch.int64)

			features = features.to(device)

			data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))
			optimizer.zero_grad()

			output = self.model(features)

			loss = loss_fn(output, call_label)
			loss.backward()

			optimizer.step()

			epoch_loss.update(loss)

			if call_label is not None:
				prediction = torch.argmax(output.data, dim=1)

				if self.num_classes > 2:
					top_k = torch.topk(output.data, k=self.topK, dim=1, largest=True, sorted=True)
					top_k_values, top_k_indices = top_k
					topk_acc.update(labels=call_label, predictions_topk=top_k_indices)
					topk_acc_rew.update(labels=call_label, predictions_topk=top_k_indices)

				if isinstance(metrics, list):
					for metric in metrics:
						metric.update(call_label, prediction)
				else:
					for metric in metrics.values():
						metric.update(call_label, prediction)

				if cm_train is not None:
					cm_train.add(output=prediction, target=call_label)

			if i == 0:
				self.write_summaries(
					features=features,
					labels=call_label,
					prediction=prediction,
					file_names=label["file_name"],
					epoch=epoch,
					phase="train",
					label_dict_swapped=self.label_dict_swapped
				)
			start_data_loading = time.time()

		self.write_scalar_summaries_logs(
			loss=epoch_loss.get(),
			metrics=metrics,
			lr=optimizer.param_groups[0]["lr"],
			epoch_time=time.time() - epoch_start,
			data_loading_time=data_loading_time.get(),
			epoch=epoch,
			phase="train",
		)

		if call_label is not None and cm_train is not None:
			confusion_matrix_raw = cm_train.confusion.clone()
			confusion_matrix_norm = cm_train.value()

			recall, precision, f1, acc, tp, fp, fn = calculate_metrics(confusion_matrix_raw)

			self.logger.debug(f"All Classes ({self.num_classes}) | Recall: {recall}|Precision: {precision}|F1: {f1}|Acc: {acc}|TP: {tp}|FP: {fp}|FN: {fn}")

			for idx in range(confusion_matrix_raw.shape[0]):
				self.logger.debug(f"Class {train_loader.dataset.get_label_string(idx)} |recall: {recall[idx]}|precision: {precision[idx]} | f1: {f1[idx]} | tp: {tp[idx]} | fp: {fp[idx]} | fn: {fn[idx]}")

			label_str = [self.label_dict_swapped.get(i) for i in range(confusion_matrix_norm.shape[0])]

			if self.num_classes > 2:
				self.logger.debug("Accuracy: {}".format(acc))
				self.logger.debug("Top-" + str(self.topK) + " Weighted Accuracy: {}".format(topk_acc_rew.value()))
				self.logger.debug("Top-" + str(self.topK) + " Unweighted Accuracy: {}".format(topk_acc.value()))

			self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase="train", norm=True, numbering=True)
			self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase="train", norm=True, numbering=False)
			self.write_confusion_summary(confusion_matrix_raw, label_str, epoch=epoch, phase="train", norm=False, numbering=True)

		self.writer.flush()

		return epoch_loss.get(), acc, topk_acc_rew.value(), topk_acc.value()

	""" 
	Validation/Testing using pre-extracted validation/test data, given loss function and respective metrics.
	The parameter 'phase' is used to switch between validation and test
	"""
	def test_epoch(self, epoch, test_loader, loss_fn, metrics, device, phase="val"):
		self.logger.debug("{}|{}|start".format(phase, epoch))
		self.model.eval()

		with torch.no_grad():
			if isinstance(metrics, list):
				for metric in metrics:
					metric.reset(device)
			else:
				for metric in metrics.values():
					metric.reset(device)

			topk_acc = Topk_accuracy(device=device, k=self.topK)
			topk_acc.reset(device)

			topk_acc_rew = Topk_weighted_accuracy(device=device, k=self.topK)
			topk_acc_rew.reset(device)

			epoch_start = time.time()
			start_data_loading = epoch_start
			data_loading_time = m.Sum(torch.device("cpu"))

			epoch_loss = m.Mean(device)

			cm_eval = ConfusionMeter(n_categories=self.num_classes)

			for i, (features, label) in enumerate(test_loader):

				call_label = label["label"].to(device, non_blocking=True, dtype=torch.int64)

				features = features.to(device)

				data_loading_time.update(
					torch.Tensor([(time.time() - start_data_loading)])
				)

				output = self.model(features)

				loss = loss_fn(output, call_label)
				epoch_loss.update(loss)

				prediction = None

				if call_label is not None:
					prediction = torch.argmax(output.data, dim=1)

					if self.num_classes > 2:
						top_k = torch.topk(output.data, k=self.topK, dim=1, largest=True, sorted=True)
						top_k_values, top_k_indices = top_k
						topk_acc.update(labels=call_label, predictions_topk=top_k_indices)
						topk_acc_rew.update(labels=call_label, predictions_topk=top_k_indices)

					if isinstance(metrics, list):
						for metric in metrics:
							metric.update(call_label, prediction)
					else:
						for metric in metrics.values():
							metric.update(call_label, prediction)

					if cm_eval is not None:
						cm_eval.add(prediction, call_label)

				if i == 0:
					self.write_summaries(
						features=features,
						labels=call_label,
						prediction=prediction,
						file_names=label["file_name"],
						epoch=epoch,
						phase=phase,
						label_dict_swapped=self.label_dict_swapped
					)
				start_data_loading = time.time()

		self.write_scalar_summaries_logs(
			loss=epoch_loss.get(),
			metrics=metrics,
			epoch_time=time.time() - epoch_start,
			data_loading_time=data_loading_time.get(),
			epoch=epoch,
			phase=phase,
		)

		if call_label is not None and cm_eval is not None:
			confusion_matrix_raw = cm_eval.confusion.clone()
			confusion_matrix_norm = cm_eval.value()

			recall, precision, f1, acc, tp, fp, fn = calculate_metrics(confusion_matrix_raw)

			self.logger.debug(
				f"All Classes ({self.num_classes}) | Recall: {recall}|Precision: {precision}|F1: {f1}|Acc: {acc}|TP: {tp}|FP: {fp}|FN: {fn}")

			for idx in range(confusion_matrix_raw.shape[0]):
				self.logger.debug(
					f"Class {test_loader.dataset.get_label_string(idx)} |recall: {recall[idx]}|precision: {precision[idx]} | f1: {f1[idx]} | tp: {tp[idx]} | fp: {fp[idx]} | fn: {fn[idx]}")

			label_str = [self.label_dict_swapped.get(i) for i in range(confusion_matrix_norm.shape[0])]

			if self.num_classes > 2:
				self.logger.debug("Accuracy: {}".format(acc))
				self.logger.debug("Top-" + str(self.topK) + " Weighted Accuracy: {}".format(topk_acc_rew.value()))
				self.logger.debug("Top-" + str(self.topK) + " Unweighted Accuracy: {}".format(topk_acc.value()))

			self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase=phase, norm=True, numbering=True)
			self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase=phase, norm=True, numbering=False)
			self.write_confusion_summary(confusion_matrix_raw, label_str, epoch=epoch, phase=phase, norm=False, numbering=True)

		self.writer.flush()

		return epoch_loss.get(), acc, topk_acc_rew.value(), topk_acc.value()


	"""
	Writes image summary per partition (corresponding images and predictions)
	"""
	def write_summaries(
		self,
		features,
		labels=None,
		label_dict_swapped=None,
		prediction=None,
		file_names=None,
		epoch=None,
		phase="train",
	):
		with torch.no_grad():
			self.write_img_summaries(
				features,
				labels=labels,
				prediction=prediction,
				file_names=file_names,
				epoch=epoch + 1,
				phase=phase,
				label_dict_swapped=label_dict_swapped
			)

	"""
	Writes image summary per partition with respect to the prediction output
	"""
	def write_img_summaries(
		self,
		features,
		labels=None,
		label_dict_swapped=None,
		prediction=None,
		file_names=None,
		epoch=None,
		phase="train",
	):
		with torch.no_grad():
			if file_names is not None:
				if isinstance(file_names, torch.Tensor):
					file_names = file_names.cpu().numpy()
				elif isinstance(file_names, list):
					file_names = np.asarray(file_names)
			if labels is not None and prediction is not None:
				features = features.cpu()
				labels = labels.cpu()
				prediction = prediction.cpu()

				t_i = torch.eq(prediction, labels)

				for idx in range(len(t_i)):
					if t_i[idx]:
						name_t = "true - " + str(label_dict_swapped.get(labels[idx].item())) + " as " + str((label_dict_swapped.get(prediction[idx].item())))
					else:
						name_t = "false - " + str(label_dict_swapped.get(labels[idx].item())) + " as " + str((label_dict_swapped.get(prediction[idx].item())))

					try:
						self.writer.add_image(
							tag=phase + "/" + name_t,
							img_tensor=prepare_img(
								features[idx].unsqueeze(dim=0),
								num_images=self.n_summaries,
								file_names=file_names[idx],
								_grayscale=self.grayscale,
							),
							global_step=epoch,
						)
					except ValueError:
						pass

	"""
	Writes scalar summary per partition including loss, data loading time, learning rate, (multi-class) classification accuracy, time per epoch,
	
	"""
	def write_scalar_summaries_logs(
		self,
		loss: float,
		metrics: Union[list, dict] = [],
		lr: float = None,
		epoch_time: float = None,
		data_loading_time: float = None,
		epoch=None,
		phase="train",
	):
		with torch.no_grad():
			log_str = phase
			if epoch is not None:
				log_str += "|{}".format(epoch)
			self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
			log_str += "|loss:{:0.3f}".format(loss)
			if isinstance(metrics, dict):
				for name, metric in metrics.items():
					self.writer.add_scalar(phase + "/" + name, metric.get(), epoch)
					log_str += "|{}:{:0.3f}".format(name, metric.get())
			else:
				for i, metric in enumerate(metrics):
					self.writer.add_scalar(
						phase + "/metric_" + str(i), metric.get(), epoch
					)
					log_str += "|m_{}:{:0.3f}".format(i, metric.get())
			if lr is not None:
				self.writer.add_scalar("lr", lr, epoch)
				log_str += "|lr:{:0.2e}".format(lr)
			if epoch_time is not None:
				self.writer.add_scalar(phase + "/time", epoch_time, epoch)
				log_str += "|t:{:0.1f}".format(epoch_time)
			if data_loading_time is not None:
				self.writer.add_scalar(
					phase + "/data_loading_time", data_loading_time, epoch
				)
			self.logger.info(log_str)

	"""
	Writes confusion matrix summary for training, validation, and test set 
	"""
	def write_confusion_summary(self, confusion_matrix, label_str=None, epoch=None, phase="", norm=True, numbering=True):
		with torch.no_grad():
			if phase != "":
				phase += "_"
			fig = confusion_matrix_fig(confusion_matrix, label_str=label_str, numbering=numbering)
			if norm:
				if numbering:
					cm_file = "confusion_matrix_norm/cm_numbered"
				else:
					cm_file = "confusion_matrix_norm/cm"
			else:
				if numbering:
					cm_file = "confusion_matrix_raw/cm_numbered"
				else:
					cm_file = "confusion_matrix_raw/cm"
			self.writer.add_figure(phase + cm_file, fig, epoch)
