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
* Pour generer le graphe UML, il faut installee le module ``pip install pylint``
    Il faut ensuite saisir la commande suivante:
    * ``pyreverse -A -S -f ALL -o png -p laue/ laue/``
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

from .core import (cam_to_gnomonic, cam_to_thetachi,
    dist_cosine, dist_euclidian, dist_line, gnomonic_to_cam,
    gnomonic_to_thetachi, hough, hough_reduce, inter_lines,
    thetachi_to_cam, thetachi_to_gnomonic, Transformer,
    comb2ind, ind2comb, atomic_pic_search, atomic_find_subsets,
    atomic_find_zone_axes)
from .experiment import Experiment, OrderedExperiment
from .utilities import (Recordable, read_image, create_image,
    images_to_iter, TimeCost, Lambdify, limited_imap,
    pickleable_method, prevent_generator_size, reduce_object,
    NestablePool, RecallingIterator, extract_parameters)

__all__ = [
    # laue.core
    "cam_to_gnomonic", "cam_to_thetachi", "dist_cosine", "dist_euclidian",
    "dist_line", "gnomonic_to_cam", "gnomonic_to_thetachi", "hough",
    "hough_reduce", "inter_lines", "thetachi_to_cam", "thetachi_to_gnomonic",
    "Transformer", "comb2ind", "ind2comb",
    "atomic_pic_search", "atomic_find_subsets", "atomic_find_zone_axes",

    # laue.experiment
    "Experiment", "OrderedExperiment",

    # laue.utilities
    "Recordable", "read_image", "create_image",
    "images_to_iter", "TimeCost", "Lambdify", "limited_imap",
    "pickleable_method", "prevent_generator_size", "reduce_object",
    "NestablePool", "RecallingIterator", "extract_parameters",
   ]

__pdoc__ = {"tests": False,
            "data": False}
