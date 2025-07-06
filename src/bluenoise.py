import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc


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


def poisson_disc_scipy(texture_size: int, normalized_dist_between_points: float) -> np.ndarray:
	sampler = qmc.PoissonDisk(d = 2, radius = normalized_dist_between_points)
	samples = sampler.random()
	pixel_coords = np.floor(samples * texture_size).astype(int)
	pixel_coords = np.clip(pixel_coords, 0, texture_size - 1)
	texture = np.zeros((texture_size, texture_size), dtype = np.uint8)
	for(x, y) in pixel_coords:
		texture[y, x] = 255

	return texture

def poisson_disc_scipy_pixel_dist(texture_size: int, pixel_dist_between_points: float) -> np.ndarray:
	normalized_dist = pixel_dist_between_points / texture_size

	sampler = qmc.PoissonDisk(d=2, radius=normalized_dist)
	#samples = sampler.random(20)
	samples = sampler.fill_space()

	pixel_coords = np.floor(samples * texture_size).astype(int)
	pixel_coords = np.clip(pixel_coords, 0, texture_size - 1)

	mask = np.zeros((texture_size, texture_size), dtype = bool) # mask to mark if points are poisson samples
	texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
	for (x, y) in pixel_coords:
		texture[y, x] = (255, 255, 255)
		mask[y, x] = True

	for y in range(0, texture_size):
		for x in range(0, texture_size):
			if not mask[y, x]:
				min_dist = float('inf')
				# slow fucking loop
				nearest = (0, 0)
				for (px, py) in pixel_coords:
					dist = (px - x) ** 2 + (py - y) ** 2
					if dist < min_dist:
						min_dist = dist
						nearest = (px, py)
				texture[y, x] = (nearest[0], nearest[1], 0)

	return texture

def filter_void_and_cluster_blue_noise_textures(blue_noise_texture: np.ndarray, width: int, height: int, cell_size: int) -> np.ndarray:
	output_texture = np.zeros((height, width), dtype = np.uint8)

	if (width & 0b01) != 0 or (height & 0b01) != 0:
		print("Width and height should be even, and preferably powers of 2!")
	for i in range(0, width, cell_size):
		for j in range(0, height, cell_size):
			subregion = blue_noise_texture[j : j + cell_size, i : i + cell_size]
			max_index = np.unravel_index(np.argmax(subregion), subregion.shape)
			global_idx = (j + max_index[0], i + max_index[1])
			output_texture[global_idx[0], global_idx[1]] = 255


	return output_texture