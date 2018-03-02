import numpy as np 

class pattern():
	def __init__(self, nchar=5, patternlen=5, repeat=False):
		self.nchar = nchar
		self.alphabet = map(chr, range(65, 91))

		self.chars = self.alphabet[:nchar]

		self.patternlen = patternlen
		self.repeat = 

	def reset(self):
		self.sequence = np.random.choice(self.chars, self.patternlen, replace=self.repeat)

