#!/usr/bin/env python3

"""
** Permet a une experience d'etre organisee. **
-----------------------------------------------

Chaque image de diagrame de laue a ete pris a un endroit
particulier de l'echantillon et en un point precis.
Cette classe permet de gerer simplement cela.
"""

import numpy as np

from laue.experiment.base_experiment import Experiment


__pdoc__ = {"OrderedExperiment.__getitem__": True}


class OrderedExperiment(Experiment):
    """
    ** Permet de travailler sur un lot ordonne d'images. **
    """
    def __init__(self, *args, position, time=(lambda ind: .0*ind), **kwargs):
        """
        Parameters
        ----------
        time : callable
            Fonction qui a tout indice de diagrame (int)
            Associe le temps ecoule depuis le debut de l'experience (float).
            Les indices commencent a 0 inclu.
        position : callable
            Fonction qui a tout indice de diagrame (int)
            Associe la position en x et en y du diagrame
            correspondant. Les indices commencent a 0 inclu.

        *args
            Same as ``laue.Experiment.__init__``.
        **kwargs
            Same as ``laue.Experiment.__init__``.
        """
        assert hasattr(position, "__call__"), "'position' has to be callable."
        assert hasattr(time, "__call__"), "'time' has to be a callable."
        
        Experiment.__init__(self, *args, **kwargs)

        self.time = time
        self.position = position

        # Memory attrs.
        self._next_row = 0
        self._t_min, self._t_max = None, None
        self._x_min, self._x_max = None, None
        self._y_min, self._y_max = None, None
        self._t_dict, self._x_dict, self._y_dict = {}, {}, {}
        self._index = None

    def _full_update(self, i_max=None):
        """
        ** Met a jour les tables. **

        Les elements sont calcules jusqu'a ``self[i_max]`` inclu.
        """
        if i_max is None:
            i_max = len(self._buff_diags)-1
        for current_row in range(self._next_row, i_max+1):
            self._update(current_row)
            self._next_row = current_row + 1

    def _update(self, i):
        """
        ** Ajoute un elements aux tables de correspondances. **
        """
        t, (x, y) = self.time(i), self.position(i)

        self._t_dict[t] = self._t_dict.get(t, set()) | {i}
        self._x_dict[x] = self._x_dict.get(x, set()) | {i}
        self._y_dict[y] = self._y_dict.get(y, set()) | {i}

        self._t_min = min(self._t_min, t) if self._t_min is not None else t
        self._t_max = min(self._t_max, t) if self._t_max is not None else t
        self._x_min = min(self._x_min, x) if self._x_min is not None else x
        self._x_max = min(self._x_max, x) if self._x_max is not None else x
        self._y_min = min(self._y_min, y) if self._y_min is not None else y
        self._y_max = min(self._y_max, y) if self._y_max is not None else y

    def get_shape(self):
        """
        ** Recupere les 2 dimensions x, y. **

        Returns
        -------
        x : int
            Le nombre de diagrames sur l'axe x.
        y : int
            Le nombre de diagrames pris selon l'axe y.

        Examples
        --------
        >>> from itertools import cycle
        >>> import laue
        >>> images = cycle(["laue/examples/ge_blanc.mccd"])
        >>> def get_positions(i, x_max, y_max):
        ...     i_mod = i % (x_max*y_max)
        ...     return divmod(i_mod, y_max)
        ...
        >>> experiment = laue.OrderedExperiment(images,
        ...     position=(lambda i: get_positions(i, 41, 82)))
        >>> experiment.get_shape()
        (41, 82)
        >>>
        """
        def is_first():
            """Renvoie True si on a fini le cycle."""
            x, y = self.position(self._next_row)
            return bool(self._x_dict.get(x, set()) & self._y_dict.get(y, set()))

        while not is_first():
            self._full_update(self._next_row)

        return len(self._x_dict), len(self._y_dict)

    def get_index(self):
        """
        ** Recupere la matrice des index d'une couche. **

        Returns
        -------
        np.ndarray
            La matrice 2d ayant le role d'une fonction de N**2 dans N.
            A couple de rang (x, y) associ le rang 'ravel' du diagrame.

        Examples
        --------
        >>> from itertools import cycle
        >>> import laue
        >>> images = cycle(["laue/examples/ge_blanc.mccd"])

        Cas balayage.
        >>> def get_position(i):
        ...     i_mod = i % 12
        ...     return divmod(i_mod, 4)
        ...
        >>> experiment = laue.OrderedExperiment(images, position=get_position)
        >>> experiment.get_index()
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=uint32)
        >>>

        Cas non conventionel
        >>> def get_position(i):
        ...     x = [0, 1, 1, 2, 2, 0, 2, 1, 0, 1, 0, 2]
        ...     y = [0, 2, 3, 1, 0, 2, 2, 0, 1, 1, 3, 3]
        ...     return x[i%12], y[i%12]
        ...
        >>> experiment = laue.OrderedExperiment(images, position=get_position)
        >>> experiment.get_index()
        array([[ 0,  8,  5, 10],
               [ 7,  9,  1,  2],
               [ 4,  3,  6, 11]], dtype=uint32)
        >>>
        """
        def min_inter(a, b, x_ind, y_ind):
            inter = a & b
            if self.ignore_errors:
                return min(inter, default=np.nan)
            if not inter:
                raise ValueError(f"pour la {x_ind+1}eme coordonnee sur x"
                                 f"et la {y_ind+1}eme coordonnee sur y, "
                                  "aucun numero de diagrame n'est fournis "
                                  "par la fonction ``self.position``.") from err
            return min(a & b)

        if self._index is not None:
            return self._index

        x, y = self.get_shape()
        self._index = np.array([[
                    min_inter(ind_x_set, ind_y_set, x_ind, y_ind)
                    for y_ind, ind_y_set
                    in sorted(self._y_dict.items(), key=lambda t: t[0])]
                for x_ind, ind_x_set
                in sorted(self._x_dict.items(), key=lambda t: t[0])],
            dtype=np.uint32)
        return self._index

    def __getitem__(self, item):
        """
        ** Recupere un diagrame ou un tenseur de diagrames. **

        Retourne le ou les diagrames de type ``laue.diagram.LaueDiagram``.

        Parameters
        ----------
        item
            Il faut voir cette experience comme un tableau
            numpy a 3 dimensions. La premiere est associee au temps,
            la seconde a l'axe x et la derniere a l'axe y.

        Returns
        -------
        diagrams
            * Si l'une des 3 dimensions est omise ou bien que c'est
            un slice, retourne une array numpy remplie de diagrames.
            * Si les 3 coordonnees sont entieres, le diagrame
            correspondant est renvoye.

        Examples
        --------
        >>> from itertools import cycle
        >>> import laue
        >>> images = cycle(["laue/examples/ge_blanc.mccd"])
        >>> def get_position(i):
        ...     i_mod = i % 12
        ...     return divmod(i_mod, 4)
        ...
        >>> experiment = laue.OrderedExperiment(images, position=get_position)
        >>>
        """
        if isinstance(item, tuple):
            if len(item) == 3:
                x, y, t = item
                size = np.prod(self.get_shape(), dtype=np.uint32)
                if all(isinstance(coord, int) for coord in item):
                    return super().__getitem__(self.get_index()[x, y] + t*size)

        raise NotImplementedError
