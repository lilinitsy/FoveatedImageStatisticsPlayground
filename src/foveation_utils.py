import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List

# Image should already be loaded in OpenCV
# This function takes in the radii in pixels
def guenter_foveated_rendering_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]) -> np.ndarray:
	foveated_image = image.copy()
	(height, width, _) = image.shape
	(x, y) = np.ogrid[:width, :height]

	distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

	mask_4x4 = distance > radii[1]
	downsampled_4x4 = cv2.resize(image, (width // 4, height // 4), interpolation = cv2.INTER_LINEAR)
	upsampled_4x4 = cv2.resize(downsampled_4x4, (width, height), interpolation = cv2.INTER_NEAREST)
	foveated_image[mask_4x4] = upsampled_4x4[mask_4x4]

	mask_2x2 = (distance > radii[0]) & (distance <= radii[1])
	downsampled_2x2 = cv2.resize(image, (width // 2, height // 2), interpolation = cv2.INTER_LINEAR)
	upsampled_2x2 = cv2.resize(downsampled_2x2, (width, height), interpolation = cv2.INTER_NEAREST)
	foveated_image[mask_2x2] = upsampled_2x2[mask_2x2]

	return foveated_image
	 


def guenter_foveated_rendering_px_mip(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]) -> np.ndarray:
	foveated_image = np.zeros_like(image)
	(height, width, _) = image.shape
	(x, y) = np.ogrid[:width, :height]

	distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
	downsampled_4x4 = cv2.resize(image, (width // 4, height // 4), interpolation = cv2.INTER_LINEAR)
	downsampled_2x2 = cv2.resize(image, (width // 2, height // 2), interpolation = cv2.INTER_LINEAR)

	for x in range(0, width):
		for y in range(0, height):
			distance = distances[x, y]

			if distance > radii[1]:
				downsampled_x = x // 4
				downsampled_y = y // 4
				foveated_image[x, y] = downsampled_4x4[downsampled_x, downsampled_y]
			elif distance > radii[0]:
				downsampled_x = x // 2
				downsampled_y = y // 2
				foveated_image[x, y] = downsampled_2x2[downsampled_x, downsampled_y]
			else:
				foveated_image[x, y] = image[x, y]

	return foveated_image
	 
def visualize_guenter_foveated_regions_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]):
	(height, width, _) = image.shape
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Matplotlib needs RGB

	(fig, ax) = plt.subplots(figsize=(10, 10))
	ax.imshow(image_rgb)
	ax.axis("off")

	for (radius, colour) in zip(radii, ["red", "blue"]):
		circle = plt.Circle(center, radius, color = colour, fill = False, linewidth = 2, label = f"Radius: {radius}")
		ax.add_artist(circle)

	plt.show()



def visualize_foveated_grid_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int], mask_widths: Tuple[int, int]):
	(height, width, _) = image.shape
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Matplotlib needs RGB
	(x, y) = np.ogrid[:width, :height]
	distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
	
	(fig, ax) = plt.subplots(figsize=(10, 10))
	ax.imshow(image_rgb)
	ax.axis("off")

	mask_central = distances <= radii[0] # do not draw over this
	mask_inner = (distances > radii[0]) & (distances <= radii[1])
	mask_outer = distances > radii[1]

	valid_2x2_rows = np.any(mask_inner, axis=1)
	valid_2x2_cols = np.any(mask_inner, axis=0)

	for x in np.where(valid_2x2_cols)[0][::mask_widths[0]]:
		valid_y = np.where(mask_inner[:, x] & ~mask_central[:, x])[0]
		if valid_y.size > 0:
			ax.vlines(x, valid_y[0], valid_y[-1] + 1, colors="red", linewidth = 0.5, alpha = 0.5)

	for y in np.where(valid_2x2_rows)[0][::mask_widths[0]]:
		valid_x = np.where(mask_inner[y, :] & ~mask_central[y, :])[0]
		if valid_x.size > 0:
			ax.hlines(y, valid_x[0], valid_x[-1] + 1, colors="red", linewidth = 0.5, alpha = 0.5)

	valid_4x4_rows = np.any(mask_outer, axis=1)
	valid_4x4_cols = np.any(mask_outer, axis=0)

	for x in np.where(valid_4x4_cols)[0][::mask_widths[1]]:
		valid_y = np.where(mask_outer[:, x] & ~mask_central[:, x])[0]
		if valid_y.size > 0:
			ax.vlines(x, valid_y[0], valid_y[-1] + 1, colors="blue", linewidth = 0.5, alpha = 0.5)

	for y in np.where(valid_4x4_rows)[0][::mask_widths[1]]:
		valid_x = np.where(mask_outer[y, :] & ~mask_central[y, :])[0]
		if valid_x.size > 0:
			ax.hlines(y, valid_x[0], valid_x[-1] + 1, colors="blue", linewidth = 0.5, alpha = 0.5)



	plt.show()


# Revisit the value for base_pooling_size....
def compute_base_moments(image: np.ndarray, fixation_point: Tuple[int, int], alpha: float = 0.1, base_pooling_size = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mean_texture = np.zeros_like(image, dtype = np.float32)
	variance_texture = np.zeros_like(image, dtype = np.float32)
	skew_texture = np.zeros_like(image, dtype = np.float32)

	(height, width, _) = image.shape
	(x, y) = np.ogrid[:width, :height]
	distances = np.sqrt((x - fixation_point[0]) ** 2 + (y - fixation_point[1]) ** 2)
	pooling_sizes = base_pooling_size + alpha * distances
	pooling_sizes = pooling_sizes.astype(int)
	half_pools = pooling_sizes // 2

	x_mins = np.clip(x - half_pools, 0, width - 1).flatten()
	x_maxes = np.clip(x + half_pools, 0, width - 1).flatten()
	y_mins = np.clip(y - half_pools, 0, height - 1).flatten()
	y_maxes = np.clip(y + half_pools, 0, height - 1).flatten()

	region_sizes = (x_maxes - x_mins, y_maxes - y_mins)
	regions = [
		image[x_min:x_max, y_min:y_max]
		for x_min, x_max, y_min, y_max in zip(x_mins, x_maxes, y_mins, y_maxes)
	]

	mean_texture_flat = [np.mean(region) for region in regions]
	variance_texture_flat = [np.var(region) for region in regions]
	skew_texture_flat = [
		0 if np.std(region) == 0 else np.mean((region - np.mean(region)) ** 3) / np.std(region) ** 3
		for region in regions
	]	

	mean_texture = np.array(mean_texture_flat).reshape(width, height)
	variance_texture = np.array(variance_texture_flat).reshape(width, height)
	skew_texture = np.array(skew_texture_flat).reshape(width, height)
	

	return (mean_texture, variance_texture, skew_texture)
