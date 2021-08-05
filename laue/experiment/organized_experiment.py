#!/usr/bin/env python3

"""
** Permet a une experience d'etre organisee. **
-----------------------------------------------

Chaque image de diagrame de laue a ete pris a un endroit
particulier de l'echantillon et en un point precis.
Cette classe permet de gerer simplement cela.
"""

from laue.experiment.base_experiment import Experiment

class OrganizedExperiment(Experiment):
	"""
	** Permet de travailler sur un lot ordonne d'images. **
	"""
	def __init__(self, *args, **kwargs):
		"""
		Parameters
		----------
		"""
		Experiment.__init__(self, *args, **kwargs)