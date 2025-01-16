# Based off Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
# (https://dl.acm.org/doi/pdf/10.1145/3386569.3392481)
# https://cwyman.org/papers/rtg2-weightedReservoir.pdf is more clear
# https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/main/Rendering/ReSTIRGI/GIReservoir.slang is the reservoir
#	from a later paper. https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/main/Rendering/ReSTIRGI/GIResampling.cs.slang uses it
# Reweighting happens here: https://github.com/Alegruz/Screen-Space-ReSTIR-GI/blob/6cf4456cca2f9a885ccc8d593a1f5700d14e426d/Rendering/ReSTIRGI/GIResampling.cs.slang#L457


# Another key insight from ReSTIR is a raycast direction for the AO visibility, not the lighting result.


import random
from typing import Dict, List, Tuple

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

	def update(self, sample_x, sample_weight, confidence = 0.5):
		self.num_elements_seen = self.num_elements_seen + 1
		self.weighted_sum = self.weighted_sum + sample_weight
		self.confidence = self.confidence + confidence

		for k in range(0, self.size):
			randnum = random.random()
			if self.sample is None:
				self.sample = sample_x
				self.sample_weight = sample_weight
			elif randnum < (sample_weight / self.weighted_sum):
				self.sample = sample_x # output sample for this k
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
	r.weighted_sum = reservoir1.weighted_sum + reservoir2.weighted_sum
	r.num_elements_seen = reservoir1.num_elements_seen + reservoir2.num_elements_seen
	# What's s.W in Alg4?

	return r


def reservoir_spatial_reuse(input_reservoirs, neighbour_width, width, height):
	reservoirs_spatial_reuse = [[input_reservoirs[x][y] for y in range(width)] for x in range(height)]

	for x in range(neighbour_width, width - neighbour_width):
		for y in range(neighbour_width, height - neighbour_width):
			neighbour_offset = (random.randint(-neighbour_width, neighbour_width), random.randint(-neighbour_width, neighbour_width))
			neighbour_reservoir = input_reservoirs[x + neighbour_offset[0]][y + neighbour_offset[1]]
			reservoirs_spatial_reuse[x][y] = combine_reservoirs((x, y), 0.5, input_reservoirs[x][y], neighbour_reservoir)

	return reservoirs_spatial_reuse