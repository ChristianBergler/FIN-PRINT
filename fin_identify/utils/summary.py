"""
Module: summary.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch

from PIL import ImageDraw
from torchvision.utils import make_grid
from torchvision import transforms

"""
Prepare given image data for tensorboard visualization
"""
def prepare_img(img, num_images=4, file_names=None, _grayscale=False):
	with torch.no_grad():
		if img.shape[0] == 0:
			raise ValueError("`img` must include at least 1 image.")

		if num_images < img.shape[0]:
			tmp = img[:num_images]
		else:
			tmp = img

		if file_names is not None:
			for i in range(tmp.shape[0]):
				try:
					pil = transforms.ToPILImage()(tmp[i]).convert("RGB")
					if _grayscale:
						pil = pil.convert("L")
					draw = ImageDraw.Draw(pil)
					draw.text(
						(2, 2),
						os.path.basename(file_names[i]),
						(255, 255, 255),
					)
					tmp[i] = pil
				except TypeError:
					pass

		tmp = make_grid(tmp, nrow=1)
		return tmp.cpu().numpy()

"""
Plot Confusion Matrix
"""
def confusion_matrix_fig(confusion, label_str=None, numbering=True):
	if isinstance(confusion, torch.Tensor):
		confusion = confusion.numpy()
	if label_str is None:
		label_str = [str(i) for i in range(confusion.shape[0])]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion, cmap="hot_r")
	fig.colorbar(cax)

	tick_size = list(range(0, confusion.shape[0], 1))

	ax.set_xticks(np.array(tick_size))
	ax.set_xticklabels(label_str, rotation=90)

	ax.set_yticks(np.array(tick_size))
	ax.set_yticklabels(label_str)

	ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

	if numbering:
		for (i, j), z in np.ndenumerate(confusion):
			ax.text(j, i, '{:0.1f}'.format(z), size='smaller', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	return fig
