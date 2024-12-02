# Based off Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting
# (https://dl.acm.org/doi/pdf/10.1145/3386569.3392481)

import random

# samples will be vec3 or vec4's?
class Reservoir:
	def __init__(self):
		self.output_sample = 0 # y in Algorithm 2
		self.weighted_sum = 0 # wsum
		self.num_samples_seen = 0 # M

	def update(self, x_sample, w_sample):
		self.weighted_sum = self.weighted_sum + w_sample
		self.num_samples_seen = self.num_samples_seen + 1

		randnum = random.random()
		if randnum < (w_sample / self.weighted_sum):
			self.output_sample = x_sample
