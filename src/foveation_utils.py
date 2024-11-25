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

	mask_4x4 = distances > radii[1]
	downsampled_4x4 = cv2.resize(image, (width // 4, height // 4), interpolation = cv2.INTER_LINEAR)

	mask_2x2 = (distances > radii[0]) & (distances <= radii[1])
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
	 
def visualize_foveated_grid_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]):
	(height, width, _) = image.shape
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Matplotlib needs RGB

	(fig, ax) = plt.subplots(figsize=(10, 10))
	ax.imshow(image_rgb)
	ax.axis("off")

	for (radius, colour) in zip(radii, ["red", "blue"]):
		circle = plt.Circle(center, radius, color = colour, fill = False, linewidth = 2, label = f"Radius: {radius}")
		ax.add_artist(circle)

	plt.show()