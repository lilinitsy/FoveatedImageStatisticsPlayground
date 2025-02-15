# Based off Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
# (https://dl.acm.org/doi/pdf/10.1145/3386569.3392481)
# https://cwyman.org/papers/rtg2-weightedReservoir.pdf is more clear
# https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/main/Rendering/ReSTIRGI/GIReservoir.slang is the reservoir
#	from a later paper. https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/main/Rendering/ReSTIRGI/GIResampling.cs.slang uses it
# Reweighting happens here: https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/6cf4456cca2f9a885ccc8d593a1f5700d14e426d/Rendering/ReSTIRGI/GIResampling.cs.slang#L457


# Another key insight from ReSTIR is a raycast direction for the AO visibility, not the lighting result.


import random
import copy
from typing import Dict, List, Tuple

import numpy as np

# samples will be vec3 or vec4's?
# Option A: store the best samples, without stochasticness? or store the stream of past 8? how does accumulation come into here?
# Option B: store 1 frame at 8 bits per colour channel ,diffs at x bits per colour channel but same resolution/

# The Reservoir should be used to store statistics so that they can be reused

class Reservoir:
	def __init__(self):
		# Store the sample's weight too?
		self.weighted_sum = 0 # wsum
		self.size = 1 # K, seems like it's basically always 1 in every use case
		self.num_elements_seen = 0
		self.sample = None
		self.sample_weight = 0
		self.confidence = 0
		self.confidence_cap = 15.0

	def update(self, sample_x, sample_weight, confidence = 0.5):
		self.num_elements_seen = self.num_elements_seen + 1
		self.weighted_sum = self.weighted_sum + sample_weight
		self.confidence = min(self.confidence + confidence, self.confidence_cap)

		for k in range(0, self.size):
			randnum = random.random()
			if self.sample is None:
				self.sample = sample_x
				self.sample_weight = sample_weight
			elif randnum < (sample_weight / self.weighted_sum):
				self.sample = sample_x # output sample for this k
				self.sample_weight = sample_weight

	def update_alt(self, sample_x, sample_weight):
		self.num_elements_seen = self.num_elements_seen + 1

		randnum = random.random()

		if randnum > self.sample_weight / (self.sample_weight + sample_weight):
			self.sample = sample_x
			self.sample_weight = sample_weight

	def to_string(self):
		print("\tsample: ", self.sample)
		print("\tsample_weight: ", self.sample_weight)
		print("\tconfidence: ", self.confidence)
		print("\tweighted sum: ", self.weighted_sum)
		print("\tnum elements seen: ", self.num_elements_seen)

# p_hat(q) is the target distribution for pixel q, but
# what should it be here? Strictly speaking, pixel_probability should be pixel_probability(r.y).
# TODO: pixel is unused here; remove later
def combine_reservoirs(pixel: Tuple[int, int], pixel_probability, reservoir1: Reservoir, reservoir2: Reservoir) -> Reservoir:
	r = Reservoir()

	# Need to revisit the PDF for this?
	# C way: r.y = random.random() * (reservoir1.weighted_sum + reservoir2.weighted_sum) <= reservoir1.weighted_sum ? reservoir1.select_element() : reservoir2.select_element()
	r = reservoir1 if random.random() * (reservoir1.weighted_sum + reservoir2.weighted_sum) <= reservoir1.weighted_sum else reservoir2
	r.confidence = reservoir1.confidence + reservoir2.confidence # Increasing the confidence of this?

	if(pixel[0] == 300 and pixel[1] == 200):
		print("r1 confidence: ", reservoir1.confidence)
		print("r2 confidence: ", reservoir2.confidence)

	# This weighted sum line might be wrong?
	r.weighted_sum = reservoir1.weighted_sum + reservoir2.weighted_sum
	
	'''
	IMPORTANT
	TODO SATURDAY 2/15
	REORG THIS AS s.W from Algorithm 4 in SpatioTemporal Reservoir Sampling

	The first term is strictly speaking (1 / phat(r.sample)) but eh...
	'''
	r.num_elements_seen = reservoir1.num_elements_seen + reservoir2.num_elements_seen
	#r.weighted_sum = (1.0 / reservoir1.sample_weight) * (1 / r.num_elements_seen * r.weighted_sum)

	# What's s.W in Alg4?

	return r




def reservoir_temporal_reuse(input_image: np.ndarray, foveation_LUT: np.ndarray, distances: np.ndarray, radii: List, adaptive_reservoirs: Reservoir, phat_current: np.ndarray, phat_prev: np.ndarray, width: int, height: int) -> Tuple[Reservoir, Reservoir]:
	current_frame_reservoirs = [[Reservoir() for _ in range(width)] for _ in range(height)]

	# Update the current frame reservoir
	for x in range(0, width):
		for y in range(0, height):
			sample = input_image[x][y]
			#sample_probability = 1 - foveation_LUT[x][y]
			#sample_probability = 1 / foveation_LUT[x][y]# - .1

			# Gaussian sample probability -- this looked much worse
			#sample_probability = np.exp(-distances[x][y] ** 2 / (2 * sigma ** 2))

			# Piecewise MIXED with gaussian? Now this seems dumb
			sample_probability = phat_current[y][x]


			current_frame_reservoirs[x][y].update(sample, sample_weight = sample_probability, confidence = np.clip(foveation_LUT[y][x], 0.01, 1.0)) # initialize reservoir weight to .5
			

			# Skip combining reservoirs for foveal pixels -- essentially, force a reset of the reservoir irrespective of history
			# TODO: We still need to think about a history cap being forced, or a confidence cap or both.
			if distances[y][x] <= radii[0]:
				adaptive_reservoirs[x][y] = current_frame_reservoirs[x][y]

			else:
				adaptive_reservoirs[x][y] = resample(adaptive_reservoirs[x][y], current_frame_reservoirs[x][y].confidence, phat_current[x][y], phat_prev[x][y])
				adaptive_reservoirs[x][y] = combine_reservoirs((x, y), sample_probability, current_frame_reservoirs[x][y], adaptive_reservoirs[x][y])

			if x == 300 and y == 200:
				print(x, y)
				print("Current frame reservoir:")
				current_frame_reservoirs[x][y].to_string()

				print("adaptive reservoir: ")
				adaptive_reservoirs[x][y].to_string()

			if x == 440 and y == 315:
				print(x, y)
				print("Current frame reservoir:")
				current_frame_reservoirs[x][y].to_string()

				print("adaptive reservoir: ")
				adaptive_reservoirs[x][y].to_string()


	return (adaptive_reservoirs, current_frame_reservoirs)






### TODO: Refactor this. Ignore this version, it's bad
def reservoir_spatial_reuse(input_reservoirs, neighbour_width, width, height):
	reservoirs_spatial_reuse = [[input_reservoirs[x][y] for y in range(width)] for x in range(height)]

	for x in range(neighbour_width, width - neighbour_width):
		for y in range(neighbour_width, height - neighbour_width):
			neighbour_offset = (random.randint(-neighbour_width, neighbour_width), random.randint(-neighbour_width, neighbour_width))
			neighbour_reservoir = input_reservoirs[x + neighbour_offset[0]][y + neighbour_offset[1]]
			reservoirs_spatial_reuse[x][y] = combine_reservoirs((x, y), 0.5, input_reservoirs[x][y], neighbour_reservoir)

	return reservoirs_spatial_reuse








# This is overwriting adaptive reservoir's confidence because it's not using input reservoir.update
def resample(input_reservoir: Reservoir, target_sample_confidence_ci, phat_current, phat_prev) -> Reservoir:
	'''
	for each M
		generate X (we don't have)
		wi = mi * phat * Wx
		Wx = (1 / phat) * wi
		mi = ci * phat / cj * phat
		so we're left with...

		wi = (ci / cj) * wi? 
	'''
	#r = Reservoir()
	r = copy.deepcopy(input_reservoir)
	wi = (target_sample_confidence_ci * phat_current) / (input_reservoir.confidence * phat_prev) * input_reservoir.sample_weight
	#wi = target_sample_confidence_ci / input_reservoir.confidence * input_reservoir.sample_weight
	r.update(input_reservoir.sample, wi, target_sample_confidence_ci)

	# The line below seems unnecessary. The wi and r.update already seems like it updates?
	#r.sample_weight = 1 / phat_current * r.weighted_sum

	return r
	


