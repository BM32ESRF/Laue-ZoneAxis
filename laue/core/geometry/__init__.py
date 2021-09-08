#!/usr/bin/env python3

"""
** Accesseur vers les methodes de ``laue.core.geometry.transformer.Transformer``. **
------------------------------------------------------------------------------------
"""

import inspect
import os

from .transformer import Transformer, comb2ind, ind2comb

__all__ = [
    "cam_to_gnomonic", "cam_to_thetachi", "dist_cosine", "dist_euclidian",
    "dist_line", "gnomonic_to_cam", "gnomonic_to_thetachi", "hough",
    "hough_reduce", "inter_lines", "thetachi_to_cam", "thetachi_to_gnomonic",
    "Transformer", "comb2ind", "ind2comb"]

__pdoc__ = {obj: ("Alias vers ``laue."
                  f"{inspect.getsourcefile(globals()[obj]).split('laue/')[-1][:-3].replace('/', '.').replace('.__init__', '')}"
                  f".{obj}``")
            for obj in __all__ if obj in globals()}
__pdoc__ = {**__pdoc__, **{f"{cl}.{meth}": False
            for cl in __all__ if globals().get(cl, None).__class__.__name__ == "type"
            for meth in globals()[cl].__dict__ if not meth.startswith("_")}}


def _global_transformer(meth_name, *args, **kwargs):
    if "global_transformer" not in globals():
        from laue.core.geometry.transformer import Transformer
        globals()["global_transformer"] = Transformer()
    return getattr(globals()["global_transformer"], meth_name)(*args, **kwargs)

def cam_to_gnomonic(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.cam_to_gnomonic``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("cam_to_gnomonic", *args, **kwargs)

def cam_to_thetachi(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.cam_to_thetachi``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("cam_to_thetachi", *args, **kwargs)

def dist_cosine(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.dist_cosine``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("dist_cosine", *args, **kwargs)

def dist_euclidian(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.dist_euclidian``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("dist_euclidian", *args, **kwargs)

def dist_line(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.dist_line``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("dist_line", *args, **kwargs)

def gnomonic_to_cam(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.gnomonic_to_cam``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("gnomonic_to_cam", *args, **kwargs)

def gnomonic_to_thetachi(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.gnomonic_to_thetachi``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("gnomonic_to_thetachi", *args, **kwargs)

def hough(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.hough``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("hough", *args, **kwargs)

def hough_reduce(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.hough_reduce``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("hough_reduce", *args, **kwargs)

def inter_lines(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.inter_lines``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("inter_lines", *args, **kwargs)

def thetachi_to_cam(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.thetachi_to_cam``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("thetachi_to_cam", *args, **kwargs)

def thetachi_to_gnomonic(*args, **kwargs):
    """ Accesseur vers la methode
    ``laue.core.geometry.transformer.Transformer.thetachi_to_gnomonic``
    d'une instance globale de
    ``laue.core.geometry.transformer.Transformer``."""
    return _global_transformer("thetachi_to_gnomonic", *args, **kwargs)
