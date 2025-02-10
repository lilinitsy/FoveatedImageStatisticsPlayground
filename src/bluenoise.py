import random
from typing import Dict, List, Tuple

import numpy as np


# TODO: https://onlinelibrary.wiley.com/doi/10.1111/cgf.14059 this may give the best possible blue noise sampling

def get_grid_pos(pt, cell_size):
	return int(pt[0] / cell_size), int(pt[1] / cell_size)

def is_valid(pt, grid, grid_w, grid_h, cell_size, radius):
	gx, gy = get_grid_pos(pt, cell_size)
	if gx < 0 or gy < 0 or gx >= grid_w or gy >= grid_h:
		return False
	search_radius = int(np.ceil(radius / cell_size)) + 1
	print("Search radius: ", search_radius)
	for dx in range(-search_radius, search_radius + 1):
		for dy in range(-search_radius, search_radius + 1):
			nx, ny = gx + dx, gy + dy
			if 0 <= nx < grid_w and 0 <= ny < grid_h:
				neighbor = grid[nx][ny]
				if neighbor is not None:
					if np.linalg.norm(np.array(pt) - np.array(neighbor)) < radius:
						return False
	return True

def poisson_disk_sampling(width, height, radius, k=30):
	cell_size = radius # / np.sqrt(2)
	grid_w, grid_h = int(width / cell_size) + 1, int(height / cell_size) + 1
	grid = [[None for _ in range(grid_h)] for _ in range(grid_w)]
	points = []
	active_list = []
	
	first_point = (random.uniform(0, width), random.uniform(0, height))
	points.append(first_point)
	active_list.append(first_point)
	gx, gy = get_grid_pos(first_point, cell_size)
	grid[gx][gy] = first_point
	
	while active_list:
		idx = random.randint(0, len(active_list) - 1)
		base = active_list[idx]
		found = False
		for _ in range(k):
			angle = random.uniform(0, 2 * np.pi)
			dist = random.uniform(radius, 2 * radius)
			candidate = (base[0] + np.cos(angle) * dist, base[1] + np.sin(angle) * dist)
			if 0 <= candidate[0] < width and 0 <= candidate[1] < height and is_valid(candidate, grid, grid_w, grid_h, cell_size, radius):
				points.append(candidate)
				active_list.append(candidate)
				gx, gy = get_grid_pos(candidate, cell_size)
				grid[gx][gy] = candidate
				found = True
				break
		if not found:
			active_list.pop(idx)
	
	return points



# This doesn't work
def generate_blue_noise_texture(width, height, cell_size):
	radius = cell_size * 1.5  # Adjust spacing to avoid clustering
	samples = poisson_disk_sampling(width, height, radius)
	
	texture = np.zeros((height, width), dtype=np.uint8)
	for x, y in samples:
		bx, by = int(x // cell_size) * cell_size, int(y // cell_size) * cell_size
		texture[by:by+cell_size, bx:bx+cell_size] = 255  # Mark sampled cells
	
	return texture




def filter_void_and_cluster_blue_noise_textures(blue_noise_texture: np.ndarray, width: int, height: int, cell_size: int) -> np.ndarray:
	output_texture = np.zeros((height, width), dtype = np.uint8)

	if (width & 0b01) != 0 or (height & 0b01) != 0:
		print("Width and height should be even, and preferably powers of 2!")
	for i in range(0, width, cell_size):
		for j in range(0, height, cell_size):
			subregion = blue_noise_texture[j : j + cell_size, i : i + cell_size]
			print("Subregion: ", subregion)
			max_index = np.unravel_index(np.argmax(subregion), subregion.shape)
			global_idx = (j + max_index[0], i + max_index[1])
			output_texture[global_idx[0], global_idx[1]] = 255


	return output_texture