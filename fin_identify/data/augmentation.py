"""
Module: augmentation.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

"""
Adding uniform-distributed random noise to an image
"""
def add_random_noise(img, factor_low=1, factor_high=1.4):
	# add random noise
	rand_num = random.uniform(factor_low, factor_high)
	original = np.asarray(img)
	noisy1 = original + rand_num * original.std() * np.random.random(original.shape)
	noisy1 = Image.fromarray(noisy1.astype(np.uint8), mode='RGB')
	return noisy1

"""
Rotates an image by a given degree
"""
def rotate(img, degree=25):
	random_degree = random.randint(-degree, degree+1)
	out = img.rotate(random_degree)
	return out

"""
Random uniform-distributed Gaussian blurring applied to a given input image
"""
def gaussian_blur(img, min_radius = 2 , max_radius = 5):
	random_radius = random.uniform(min_radius, max_radius)
	im2 = img.filter(ImageFilter.GaussianBlur(random_radius))
	return im2

"""
Random uniform-distributed brighten or darken of a given input image
"""
def brighten_darken(img, min_factor=0.5, max_factor=1.8):
	random_factor = random.uniform(min_factor, max_factor)
	brightened = ImageEnhance.Brightness(img).enhance(random_factor)
	return brightened

"""
Random color change of a given input image
"""
def random_color_change(img):
	rand_num = np.random.randint(0, 5)
	r, g, b = img.split()

	if rand_num == 0:
		im_split = Image.merge("RGB", (b, g, r))
	elif rand_num == 1:
		im_split = Image.merge("RGB", (r, b, g))
	elif rand_num == 2:
		im_split = Image.merge("RGB", (b, r, g))
	elif rand_num == 3:
		im_split = Image.merge("RGB", (g, r, b))
	else:
		im_split = Image.merge("RGB", (g, b, r))

	return im_split

"""
Mirroring of a given input image
"""
def mirror(img):
	return ImageOps.mirror(img)

"""
Edge enhancement of a given input image
"""
def enhance_edges(img):
	return img.filter(ImageFilter.EDGE_ENHANCE_MORE)

"""
Image sharpening
"""
def sharpen(img):
	return img.filter(ImageFilter.SHARPEN)

"""
Augmentation function which combines all specific image augmentation techniques. For each image a randomly selected number of consecutive augmentation steps 
is selected. At least 1 augmentation up to a given max number of augmentation steps. Out of the entire augmentation pool this random number is used to choose random
functions which will be applied one after another until the final augmented image is returned.
"""
def augment(original, max_augmentations=1):
	augmented = original
	functions = [add_random_noise, rotate, gaussian_blur, mirror, enhance_edges, sharpen, brighten_darken, random_color_change]

	random.shuffle(functions)

	num_augmentations = random.randint(1, max_augmentations)

	aug_str = ""
	for i in range(num_augmentations):
		if i == len(functions):
			break
		rand_fun_idx = random.randint(0, len(functions) - 1)
		fun = functions[rand_fun_idx]

		augmented = fun(augmented)
		aug_str += f"augmentation {i+1}: {fun.__name__}|"

	return augmented
