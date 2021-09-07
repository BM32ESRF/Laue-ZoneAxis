#!/usr/bin/env python3

"""
** Implemente les classes mere abstraite, point d'entree. **
------------------------------------------------------------

C'est ici que sont implementees les classes permetant
de manipuler un lot d'images de diagrame de Laue.
"""

from .base_experiment import Experiment
from .ordered_experiment import OrderedExperiment

__all__ = ["Experiment", "OrderedExperiment"]
