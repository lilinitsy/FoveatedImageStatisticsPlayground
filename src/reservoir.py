# Based off Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
# (https://dl.acm.org/doi/pdf/10.1145/3386569.3392481)
# https://cwyman.org/papers/rtg2-weightedReservoir.pdf is more clear

import random
from typing import Dict, List, Tuple

# samples will be vec3 or vec4's?
# Option A: store the best samples, without stochasticness? or store the stream of past 8? how does accumulation come into here?
# Option B: store 1 frame at 8 bits per colour channel ,diffs at x bits per colour channel but same resolution/

# The Reservoir should be used to store statistics so that they can be reused

class Reservoir:
	def __init__(self, size = 1):
		self.weighted_sum = 0 # wsum
		self.size = size # K
		self.num_elements_seen = 0
		self.reservoir = []

		for k in range(0, self.size):
			self.reservoir.append(0)

	# The below function definition could be necessary.
	# Simpler if it's not and can just upload some luminance value or something?
	#def update(self, sample: Tuple[Union[float, np.ndarray], float]):
	

	def update(self, sample_x, sample_weight):
		self.num_elements_seen = self.num_elements_seen + 1
		self.weighted_sum = self.weighted_sum + sample_weight

		for k in range(0, self.size):
			randnum = random.random()
			if randnum < (sample_weight / self.weighted_sum):
				self.reservoir[k] = sample_x # output sample for this k

	# What's a good selection strategy for k > 1?
	def select_element(self):
		return self.reservoir[-1] # return last added element for now...


# p_hat(q) is the target distribution for pixel q, but
# what should it be here? Strictly speaking, pixel_probability should be pixel_probability(r.y).
# TODO: pixel is unused here; remove later
def combine_reservoirs(pixel, pixel_probability, reservoirs: List[Reservoir]) -> Reservoir:
	r = Reservoir()
	num_elements_seen = 0
	for reservoir in reservoirs:
		sample = reservoir.select_element()
		r.update(sample, pixel_probability * reservoir.weighted_sum * reservoir.num_elements_seen) # TODO: Revisit the PDF
		num_elements_seen = num_elements_seen + reservoir.num_elements_seen
	r.num_elements_seen = num_elements_seen # Update this because the standard update function doesn't do it properly
	# What's s.W in Alg4?

