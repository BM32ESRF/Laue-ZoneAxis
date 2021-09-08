#!/usr/bin/env python3

"""
** Permet de manipuler un lot de diagrammes de laue. **
-------------------------------------------------------

Les classes principales sont organisees de la facon suivante:

.. figure:: /home/robin/documents/stages/esrf/laue_code/uml.png

Toutes les conventions et noms de variables respectent la figure suivante:

.. figure:: /home/robin/documents/stages/esrf/laue_code/geometry.jpg

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
    * ``cd laue/``
    * ``pyreverse -A -f OTHER -o png ./experiment/ordered_experiment.py
        ./diagram.py ./spot.py ./zone_axis.py ./core/geometry/transformer.py``
* A la premiere execution, les equations sont compilees, ce qui peut metre
    plusieurs disaines de minutes. Soyez patients!

Examples
--------

utilisation minimaliste
>>> import laue
>>> image = "laue/examples/ge_blanc.mccd"
>>> experiment = laue.Experiment(image)
>>>
>>> experiment
Experiment('laue/examples')
>>>
>>> for diag in experiment:
...     print(type(diag))
...
<class 'laue.diagram.LaueDiagram'>
>>>
"""

import inspect

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


__pdoc__ = {obj: ("Alias vers ``laue."
                  f"{inspect.getsourcefile(globals()[obj]).split('laue/')[-1][:-3].replace('/', '.').replace('.__init__', '')}"
                  f".{obj}``")
            for obj in __all__}
__pdoc__ = {**__pdoc__, **{f"{cl}.{meth}": False
            for cl in __all__ if globals()[cl].__class__.__name__ == "type"
            for meth in globals()[cl].__dict__ if not meth.startswith("_")}}
__pdoc__["tests"] = False
__pdoc__["data"] = False
