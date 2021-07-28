#!/usr/bin/env python3

"""
** Permet de manipuler un lot de diagrammes de laue. **
------------------------------------------------------

Notes
-----
* Pour effectuer les bancs de tests, il faut installer le module ``pip install pytest``.
    Il faut ensuite saisir la commande suivante:
    * ``clear && pytest --disable-warnings --doctest-modules laue/
        && pytest -v --exitfirst laue/tools/tests.py && cat tests_results.txt``
* Pour generer la documentation, il faut installer le module ``pip install pdoc3``.
    Il faut ensuite saisir la commande suivante:
    * ``pdoc3 laue/ -c latex_math=True --force --html``

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
Experiment(verbose=False, max_space=5, threshold=5.1)
>>>

recuperation des diagrammes
>>> for diag in experiment:
...     print(type(diag))
...
<class 'laue.diagram.LaueDiagram'>
>>>

"""

# Pour genereer la documentation, il faut taper dans un terminal:
# $ pdoc3 laue/ -c latex_math=True --force --html
# Pour faire passer les tests, il faut taper:
# $ python3 -m pytest --doctest-modules laue && python3 -m pytest laue/tools/tests.py

__all__ = ["Experiment"]
__pdoc__ = {"Experiment.__getitem__": True,
            "Experiment.__iter__": True,
            "Experiment.__len__": True,
            "tools.fork_lambdify": False}

from laue.experiment.base_experiment import Experiment
