# Based off Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
# (https://dl.acm.org/doi/pdf/10.1145/3386569.3392481)

import random

# samples will be vec3 or vec4's?
class Reservoir:
	def __init__(self, size = 1):
		self.weighted_sum = 0 # wsum
		self.size = 1 # K
		self.reservoir = []

		for k in range(0, self.size):
			self.reservoir.append(0)

	# The below function definition could be necessary.
	# Simpler if it's not and can just upload some luminance value or something?
	#def update(self, sample: Tuple[Union[float, np.ndarray], float]):


	def update(self, sample_x, sample_weight):
		self.weighted_sum = self.weighted_sum + sample_weight

		for k in range(0, len(self.reservoir)):
			randnum = random.random()
			if randnum < (sample_weight / self.weighted_sum):
				self.reservoir[k] = sample_x # output sample for this k
