#!/usr/bin/env python3

"""
** Permet de manipuler un lot de diagrammes de laue. **
-------------------------------------------------------

Notes
-----
* Pour effectuer les bancs de tests, il faut installer le module ``pip install pytest``.
    Il faut ensuite saisir la commande suivante:
    * ``clear && python -m pytest --doctest-modules laue/
        && python -m pytest -vv --exitfirst laue/tests.py && cat tests_results.txt``
* Pour generer la documentation, il faut installer le module ``pip install pdoc3``.
    Il faut ensuite saisir la commande suivante:
    * ``pdoc3 laue/ -c latex_math=True --force --html``
* A la premiere execution, les equations sont compilees, ce qui peut metre
    plusieurs disaines de minutes. Soyez patients!

Examples
--------

preparation
>>> import laue
>>>

creation d'une experience
>>> image = "laue/examples/ge_blanc.mccd"
>>> experiment = laue.Experiment(image)
>>>
>>> experiment
Experiment('laue/examples')
>>>

recuperation des diagrammes
>>> for diag in experiment:
...     print(type(diag))
...
<class 'laue.diagram.LaueDiagram'>
>>>
"""

__all__ = ["Experiment", "OrderedExperiment", "geometry",
           "atomic_pic_search", "atomic_find_zone_axes", "atomic_find_subsets"]
__pdoc__ = {"tests": False,
            "data": False,
            "Experiment.__getitem__": True,
            "Experiment.__iter__": True,
            "Experiment.__len__": True,
            "OrderedExperiment.__getitem__": True}

from laue.experiment.base_experiment import Experiment
from laue.experiment.ordered_experiment import OrderedExperiment
from laue.core import geometry
from laue.core.pic_search import atomic_pic_search
from laue.core.zone_axes import atomic_find_zone_axes
from laue.core.subsets import atomic_find_subsets
