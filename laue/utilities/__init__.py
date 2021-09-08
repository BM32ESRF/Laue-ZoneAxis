#!/usr/bin/env python3

"""
** Outils en tout genre. **
---------------------------

Toute la gestion du contexte et des fonctionalite
qui ne sont pas des fonctionnalitee de coeur sont
codee ici.
"""

import inspect

from .data_consistency import Recordable
from .image import read_image, create_image, images_to_iter
from .lambdify import TimeCost, Lambdify
from .multi_core import (limited_imap, pickleable_method,
    prevent_generator_size, reduce_object, NestablePool,
    RecallingIterator)
from .parsing import extract_parameters

__all__ = [
    "Recordable",
    "read_image", "create_image", "images_to_iter",
    "TimeCost", "Lambdify",
    "limited_imap", "pickleable_method", "prevent_generator_size",
    "reduce_object", "NestablePool", "RecallingIterator",
    "extract_parameters"]

__pdoc__ = {obj: ("Alias vers ``laue."
                  f"{inspect.getsourcefile(globals()[obj]).split('laue/')[-1][:-3].replace('/', '.').replace('.__init__', '')}"
                  f".{obj}``")
            for obj in __all__}
__pdoc__ = {**__pdoc__, **{f"{cl}.{meth}": False
            for cl in __all__ if globals()[cl].__class__.__name__ == "type"
            for meth in globals()[cl].__dict__ if not meth.startswith("_")}}
