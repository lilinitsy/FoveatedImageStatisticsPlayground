import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from reservoir import Reservoir

# Image should already be loaded in OpenCV
# This function takes in the radii in pixels
def guenter_foveated_rendering_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]) -> np.ndarray:
	foveated_image = image.copy()
	(width, height, _) = image.shape
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
	(width, height, _) = image.shape
	(x, y) = np.ogrid[:width, :height]

	distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
	downsampled_4x4 = cv2.resize(image, (width // 4, height // 4), interpolation = cv2.INTER_LINEAR)
	downsampled_2x2 = cv2.resize(image, (width // 2, height // 2), interpolation = cv2.INTER_LINEAR)

	for x in range(0, width):
		for y in range(0, height):
			distance = distances[y, x] # Numpy uses row-major and somehow this means has to be [y, x]?

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
	(width, height, _) = image.shape
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Matplotlib needs RGB
	(x, y) = np.ogrid[:width, :height]
	distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
	
	(fig, ax) = plt.subplots(figsize=(10, 10))
	ax.imshow(image_rgb)
	ax.axis("off")

	mask_central = distances <= radii[0]
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

	(width, height, _) = image.shape
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

def visualize_base_moments(mean_texture: np.ndarray, variance_texture: np.ndarray, skew_texture: np.ndarray) -> None:
	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	
	# Mean texture visualization
	axs[0].imshow(mean_texture, cmap = 'gray')
	axs[0].set_title("Mean base moment")
	axs[0].axis("off")

	# Variance texture visualization
	axs[1].imshow(variance_texture, cmap = 'gray')
	axs[1].set_title(f"Variance base moment")
	axs[1].axis("off")

	axs[2].imshow(skew_texture, cmap = 'gray')
	axs[2].set_title("Skew base moment")
	axs[2].axis("off")
	
	plt.tight_layout()
	plt.show()





# Beyond Blur pyramids are made from here: https://github.com/kaanaksit/odak/blob/196a8aa9217fa52f843c31fd9e613b64a7bd904f/odak/learn/perception/spatial_steerable_pyramid.py#L104
def compute_basemoments_gaussian_pyramids(mean_texture: np.ndarray, variance_texture: np.ndarray, skew_texture: np.ndarray, num_levels: int = 5) -> Dict[str, List[np.ndarray]]:
	gaussian_pyramids = {
		'mean': [mean_texture],
		'variance': [variance_texture],
		'skew': [skew_texture]
	}

	for i in range(1, num_levels):
		gaussian_pyramids['mean'].append(cv2.pyrDown(gaussian_pyramids['mean'][i - 1]))
		gaussian_pyramids['variance'].append(cv2.pyrDown(gaussian_pyramids['variance'][i - 1]))
		gaussian_pyramids['skew'].append(cv2.pyrDown(gaussian_pyramids['skew'][i - 1]))

	return gaussian_pyramids


def compute_basemoments_laplacian_pyramids(gaussian_pyramids: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
	laplacian_pyramids = {
		'mean': [],
		'variance': [],
		'skew': []
    }

	
	for moment in gaussian_pyramids.keys():
		num_levels = len(gaussian_pyramids[moment])
		for i in range(num_levels - 1):
			size = (gaussian_pyramids[moment][i].shape[1], gaussian_pyramids[moment][i].shape[0])
			expanded = cv2.pyrUp(gaussian_pyramids[moment][i+1], dstsize=size)
			laplacian = cv2.subtract(gaussian_pyramids[moment][i], expanded)
			laplacian_pyramids[moment].append(laplacian)
		laplacian_pyramids[moment].append(gaussian_pyramids[moment][-1])

	return laplacian_pyramids


def basic_spatial_accumulation_without_pyramids(image: np.ndarray, history_buffer: np.ndarray, alpha = 0.2) -> np.ndarray:
		return (1 - alpha) * image.astype(np.float32) + alpha * history_buffer.astype(np.float32)


# LUT = LookUp Table
def make_foveation_lookup_table(distances: np.ndarray, thresholds: List) -> np.ndarray:
	# Map the distance from center to pyramid levels
	return np.digitize(distances, thresholds, right = True) # Right = true, bins increasing, bins[i - 1] < x <= bins[i]
