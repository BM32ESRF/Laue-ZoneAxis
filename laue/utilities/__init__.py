#!/usr/bin/env python3

"""
** Outils en tout genre. **
---------------------------

Toute la gestion du contexte et des fonctionalite
qui ne sont pas des fonctionnalitee de coeur sont
codee ici.
"""

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
