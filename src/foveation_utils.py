import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List

# Image should already be loaded in OpenCV
# This function takes in the radii in pixels
def guenter_foveated_rendering_px(image: np.ndarray, center: Tuple[int, int], radii: Tuple[int, int]) -> np.ndarray:
	foveated_image: np.ndarray = image.copy()
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
	 