"""
Module: metrics.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

import torch
import numpy as np

"""
Divides two tensors element-wise, returning 0 if the denominator is <= 0.
"""
def _safe_div(numerator, denominator):
	t = torch.div(numerator, denominator)
	condition = torch.gt(denominator, float(0))
	return torch.where(condition, t, torch.zeros_like(t))

"""
Defines a counter for for various metrics (e.g. confusion matrix values).
"""
def _count_condition(condition, weights):
	with torch.no_grad():
		if weights is not None:
			if torch.is_tensor(weights):
				weights = weights.float()
			condition = torch.mul(condition.float(), weights)
		return condition.sum().item()

"""
Get weighting factors for any type of k while each weight factor decreases by a factor of 2 with respect to the previous one
"""
def get_weighting_factors(k):
	weight_factors = dict()
	for index in range(0, k):
		if index == 0:
			weight_factors[index] = 1
		else:
			weight_factors[index] = weight_factors.get(index-1)/2
	return weight_factors

"""
Based on a given (multi-class) confusion matrix the respective class-specific metrics are calculated (recall, precision, F1, accuracy)
"""
def calculate_metrics(conf_matrix):
	conf_matrix = conf_matrix.cpu().numpy()
	matrix_sum = np.sum(conf_matrix)
	tp = np.diag(conf_matrix)
	tp_sum = np.sum(tp)
	acc = tp_sum / matrix_sum if matrix_sum else tp_sum

	#rows dim=1 column dim=0 (ground truth = row, hypothesis = column)
	fn = np.sum(conf_matrix, axis=1) - tp
	fp = np.sum(conf_matrix, axis=0) - tp

	recall_denom = np.sum(conf_matrix, axis=1)
	recall = np.divide(tp, recall_denom, out=np.zeros_like(tp), where=recall_denom != 0)

	prec_denom = np.sum(conf_matrix, axis=0)
	precision = np.divide(tp, prec_denom, out=np.zeros_like(tp), where=prec_denom != 0)

	f1_numerator = (2 * recall * precision)
	f1_denominator = (recall + precision)
	f1 = np.divide(f1_numerator, f1_denominator, out=np.zeros_like(recall), where=f1_denominator != 0)

	return recall, precision, f1, acc, tp, fp, fn

"""
Define of various evaluation metrics in order to track the entire network training, validation, and testing
"""
class MetricBase:
	def __init__(self):
		pass

	def reset(self, device=None):
		self.__init__(device=device)

	def update(self):
		pass

	def _get_tensor(self):
		pass

	def get(self):
		return self._get_tensor().item()

	def __str__(self):
		return str(self.get())

	def __format__(self, spec):
		return self.get().__format__(spec)

"""
Define sum metric
"""
class Sum(MetricBase):
	def __init__(self, device=None):
		self.value = torch.zeros(1, device=device)

	def update(self, values, weights=None):
		with torch.no_grad():
			if weights is not None:
				values = torch.mul(values, weights)
			self.value += values.sum()

	def _get_tensor(self):
		return self.value

"""
Define max metric
"""
class Max(MetricBase):
	def __init__(self, device=None):
		self.value = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, values, weights=None):
		with torch.no_grad():
			if weights is not None:
				values = torch.mul(values, weights)

			tmp = values.max().float()
			if tmp > self.value:
				self.value = tmp

	def _get_tensor(self):
		return self.value

"""
Define mean metric
"""
class Mean(MetricBase):
	def __init__(self, device=None):
		self.total = torch.zeros(1, dtype=torch.float, device=device)
		self.count = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, values, weights=None):
		with torch.no_grad():
			if weights is None:
				num_values = float(values.numel())
			else:
				if torch.is_tensor(weights):
					num_values = weights.sum().float()
					weights = weights.float()
				else:
					num_values = torch.mul(values.numel(), weights).float()
				values = torch.mul(values, weights)

			self.total += values.sum().float()
			self.count += num_values

	def _get_tensor(self):
		return _safe_div(self.total, self.count).float()

"""
Define accuracy metric
"""
class Accuracy(MetricBase):
	def __init__(self, device=None):
		self.mean = Mean(device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			predictions = predictions.type_as(labels)
			is_correct = torch.eq(labels, predictions).float()
			self.mean.update(is_correct, weights)

	def _get_tensor(self):
		return self.mean._get_tensor()

"""
Define true positive metric
"""
class TruePositives(MetricBase):
	def __init__(self, device=None):
		self.count = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			predictions = predictions.type_as(labels)
			is_true = torch.eq(labels, True)
			is_positive = torch.eq(predictions, True)
			condition = torch.mul(is_true, is_positive)
			self.count += _count_condition(condition, weights)
			print(self.count)

	def _get_tensor(self):
		return self.count

"""
Define false positive metric
"""
class FalsePositives(MetricBase):
	def __init__(self, device=None):
		self.count = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			predictions = predictions.type_as(labels)
			is_false = torch.eq(labels, False)
			is_positive = torch.eq(predictions, True)
			condition = torch.mul(is_false, is_positive)
			self.count += _count_condition(condition, weights)

	def _get_tensor(self):
		return self.count

"""
Define true negative metric
"""
class TrueNegatives(MetricBase):
	def __init__(self, device=None):
		self.count = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			predictions = predictions.type_as(labels)
			is_false = torch.eq(labels, False)
			is_negative = torch.eq(predictions, False)
			condition = torch.mul(is_false, is_negative)
			self.count += _count_condition(condition, weights)

	def _get_tensor(self):
		return self.count

"""
Define false negative metric
"""
class FalseNegatives(MetricBase):
	def __init__(self, device=None):
		self.count = torch.zeros(1, dtype=torch.float, device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			predictions = predictions.type_as(labels)
			is_true = torch.eq(labels, True)
			is_negative = torch.eq(predictions, False)
			condition = torch.mul(is_true, is_negative)
			self.count += _count_condition(condition, weights)

	def _get_tensor(self):
		return self.count

"""
Define precision metric
"""
class Precision(MetricBase):
	def __init__(self, device=None):
		self.tp = TruePositives(device=device)
		self.fp = FalsePositives(device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			self.tp.update(labels, predictions, weights)
			self.fp.update(labels, predictions, weights)

	def _get_tensor(self):
		pred_p = self.tp._get_tensor() + self.fp._get_tensor()
		return torch.where(
			torch.gt(pred_p, 0),
			torch.div(self.tp._get_tensor(), pred_p),
			torch.zeros_like(pred_p),
		)

"""
Define TPR/recall metric
"""
class Recall(MetricBase):
	def __init__(self, device=None):
		self.tp = TruePositives(device=device)
		self.fn = FalseNegatives(device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			self.tp.update(labels, predictions, weights)
			self.fn.update(labels, predictions, weights)

	def _get_tensor(self):
		p = self.tp._get_tensor() + self.fn._get_tensor()
		return torch.where(
			torch.gt(p, 0), torch.div(self.tp._get_tensor(), p), torch.zeros_like(p)
		)

"""
Define FPR metric
"""
class FPR(MetricBase):
	def __init__(self, device=None):
		self.fp = FalsePositives(device=device)
		self.tn = TrueNegatives(device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			self.fp.update(labels, predictions, weights)
			self.tn.update(labels, predictions, weights)

	def _get_tensor(self):
		n = self.fp._get_tensor() + self.tn._get_tensor()
		return torch.where(
			torch.gt(n, 0), torch.div(self.fp._get_tensor(), n), torch.zeros_like(n)
		)

"""
Define F1Score metric
"""
class F1Score(MetricBase):
	def __init__(self, device=None):
		self.pr = Precision(device=device)
		self.re = Recall(device=device)

	def update(self, labels, predictions, weights=None):
		with torch.no_grad():
			self.pr.update(labels, predictions, weights)
			self.re.update(labels, predictions, weights)

	def _get_tensor(self):
		s = self.pr._get_tensor() + self.re._get_tensor()
		return torch.where(
			torch.gt(s, 0),
			torch.div(2 * self.pr._get_tensor() * self.re._get_tensor(), s),
			torch.zeros_like(s),
		)

"""
Define Top-K Accuracy, considering a hypothesis as correct if the correct prediction is within the Top-K model predictions
"""
class Topk_accuracy(MetricBase):
	def __init__(self, device=None, k=3):
		self.count = torch.zeros(1, dtype=torch.float, device=device)
		self.total = torch.zeros(1, dtype=torch.float, device=device)
		self.topk_acc = torch.zeros(1, dtype=torch.float, device=device)
		self.k = k

	def update(self, labels, predictions_topk, weights=None):
		with torch.no_grad():
			for label, pred_topk in zip(labels, predictions_topk):
				self.total += 1

				if label in pred_topk:
					self.count += 1

			self.topk_acc = _safe_div(self.count, self.total)

	def value(self):
		return self.topk_acc.item()

"""
Define Top-K Weighted Accuracy, considering a hypothesis as correct if the correct prediction is within the Top-K model predictions
however according to the position (1st rank, 2nd rank, etc.) different weights will be applied for a final accuracy scoring.
"""
class Topk_weighted_accuracy(MetricBase):
	def __init__(self, device=None, k=3):
		self.count = torch.zeros(1, dtype=torch.float, device=device)
		self.total = torch.zeros(1, dtype=torch.float, device=device)
		self.topk_acc = torch.zeros(1, dtype=torch.float, device=device)
		self.k = k
		self.weight_factors = get_weighting_factors(self.k)

	def update(self, labels, predictions_topk, weights=None):
		with torch.no_grad():

			for label, pred_topk in zip(labels, predictions_topk):
				self.total += 1
				label = label.cpu().numpy()
				pred_topk = pred_topk.cpu().numpy()
				if label in pred_topk:
					for idx in range(len(pred_topk)):
						if pred_topk[idx] == label:
							self.count += self.weight_factors.get(idx)

			self.topk_acc = _safe_div(self.count, self.total)

	def value(self):
		return self.topk_acc.item()




