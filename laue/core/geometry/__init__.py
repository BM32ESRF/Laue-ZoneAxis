#!/usr/bin/env python3

"""
** Accesseur vers les methodes de ``laue.core.geometry.transformer.Transformer``. **
-------------------------------------------------------------------------------
"""

from .transformer import Transformer, comb2ind, ind2comb

__all__ = [
    "cam_to_gnomonic", "cam_to_thetachi", "dist_cosine", "dist_euclidian",
    "dist_line", "gnomonic_to_cam", "gnomonic_to_thetachi", "hough",
    "hough_reduce", "inter_lines", "thetachi_to_cam", "thetachi_to_gnomonic",
    "Transformer", "comb2ind", "ind2comb"]

def _global_transformer(meth_name, *args, **kwargs):
    if "global_transformer" not in globals():
        from laue.core.geometry.transformer import Transformer
        globals()["global_transformer"] = Transformer()
    return getattr(globals()["global_transformer"], meth_name)(*args, **kwargs)

def cam_to_gnomonic(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.cam_to_gnomonic``."""
    return _global_transformer("cam_to_gnomonic", *args, **kwargs)

def cam_to_thetachi(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.cam_to_thetachi``."""
    return _global_transformer("cam_to_thetachi", *args, **kwargs)

def dist_cosine(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.dist_cosine``."""
    return _global_transformer("dist_cosine", *args, **kwargs)

def dist_euclidian(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.dist_euclidian``."""
    return _global_transformer("dist_euclidian", *args, **kwargs)

def dist_line(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.dist_line``."""
    return _global_transformer("dist_line", *args, **kwargs)

def gnomonic_to_cam(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.gnomonic_to_cam``."""
    return _global_transformer("gnomonic_to_cam", *args, **kwargs)

def gnomonic_to_thetachi(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.gnomonic_to_thetachi``."""
    return _global_transformer("gnomonic_to_thetachi", *args, **kwargs)

def hough(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.hough``."""
    return _global_transformer("hough", *args, **kwargs)

def hough_reduce(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.hough_reduce``."""
    return _global_transformer("hough_reduce", *args, **kwargs)

def inter_lines(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.inter_lines``."""
    return _global_transformer("inter_lines", *args, **kwargs)

def thetachi_to_cam(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.thetachi_to_cam``."""
    return _global_transformer("thetachi_to_cam", *args, **kwargs)

def thetachi_to_gnomonic(*args, **kwargs):
    """See ``laue.core.geometry.transformer.Transformer.thetachi_to_gnomonic``."""
    return _global_transformer("thetachi_to_gnomonic", *args, **kwargs)
