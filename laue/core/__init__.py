#!/usr/bin/env python3

"""
** Calculs de base de la separation de grains. **
-------------------------------------------------

C'est ici que sont effectuees les gros calculs. La mise en forme
et le contexte n'est pas gere ici.
"""

import inspect

from .geometry import (cam_to_gnomonic, cam_to_thetachi,
    dist_cosine, dist_euclidian, dist_line, gnomonic_to_cam,
    gnomonic_to_thetachi, hough, hough_reduce, inter_lines,
    Transformer, comb2ind, ind2comb,
    thetachi_to_cam, thetachi_to_gnomonic)
from .pic_search import atomic_pic_search
from .subsets import atomic_find_subsets
from .zone_axes import atomic_find_zone_axes

__all__ = [
    # geometry
    "cam_to_gnomonic", "cam_to_thetachi", "dist_cosine", "dist_euclidian",
    "dist_line", "gnomonic_to_cam", "gnomonic_to_thetachi", "hough",
    "hough_reduce", "inter_lines", "thetachi_to_cam", "thetachi_to_gnomonic",
    "Transformer", "comb2ind", "ind2comb",

    # pic_search
    "atomic_pic_search",

    # subsets
    "atomic_find_subsets",

    # zone_axes
    "atomic_find_zone_axes"]

__pdoc__ = {obj: ("Alias vers ``laue."
                  f"{inspect.getsourcefile(globals()[obj]).split('laue/')[-1][:-3].replace('/', '.').replace('.__init__', '')}"
                  f".{obj}``")
            for obj in __all__}
__pdoc__ = {**__pdoc__, **{f"{cl}.{meth}": False
            for cl in __all__ if globals()[cl].__class__.__name__ == "type"
            for meth in globals()[cl].__dict__ if not meth.startswith("_")}}
