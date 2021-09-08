#!/usr/bin/env python3

"""
** Implemente les classes mere abstraite, point d'entree. **
------------------------------------------------------------

C'est ici que sont implementees les classes permetant
de manipuler un lot d'images de diagrame de Laue.
"""

import inspect

from .base_experiment import Experiment
from .ordered_experiment import OrderedExperiment

__all__ = ["Experiment", "OrderedExperiment"]

__pdoc__ = {obj: ("Alias vers ``laue."
                  f"{inspect.getsourcefile(globals()[obj]).split('laue/')[-1][:-3].replace('/', '.').replace('.__init__', '')}"
                  f".{obj}``")
            for obj in __all__}
__pdoc__ = {**__pdoc__, **{f"{cl}.{meth}": False
            for cl in __all__ if globals()[cl].__class__.__name__ == "type"
            for meth in globals()[cl].__dict__ if not meth.startswith("_")}}
