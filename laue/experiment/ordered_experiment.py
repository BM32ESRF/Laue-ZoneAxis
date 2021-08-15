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

        Utilise uniquement la fonction ``position`` fourni a l'initialisateur.

        Returns
        -------
        x : int
            Le nombre de diagrames sur l'axe x.
        y : int
            Le nombre de diagrames pris selon l'axe y.

        Examples
        --------
        >>> import laue
        >>> def get_positions(i):
        ...     i_mod = i % 3362
        ...     return divmod(i_mod, 82)
        ...
        >>> exp = laue.OrderedExperiment((None,), position=get_positions)
        >>> exp.get_shape()
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

        Utilise uniquement la fonction ``position`` fourni a l'initialisateur.

        Returns
        -------
        np.ndarray
            La matrice 2d ayant le role d'une fonction de N**2 dans N.
            A couple de rang (x, y) associ le rang 'ravel' du diagrame.

        Examples
        --------
        >>> import laue

        Cas balayage.
        >>> def get_position(i):
        ...     i_mod = i % 12
        ...     return divmod(i_mod, 4)
        ...
        >>> experiment = laue.OrderedExperiment((None,), position=get_position)
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
        >>> experiment = laue.OrderedExperiment((None,), position=get_position)
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
            * Si seul 1 dimension est precisee, cette methode se refere a
            la methode ``laue.experiment.base_experiment.Experiment.__getitem__``.
            * Si les 3 coordonnees sont entieres, le diagrame
            correspondant est renvoye.
            * Si l'une des 3 coordonnees au moins est un slice, une array
            numpy dimension 3 est renvoyee.

        Examples
        --------
        >>> import numpy as np
        >>> import laue
        >>> images = (np.zeros(shape=(2, 2), dtype=np.uint16)
        ...     for _ in range(120)) # Image generator.
        >>> def get_position(i):
        ...     i_mod = i % 12
        ...     return divmod(i_mod, 4)
        ...
        >>> experiment = laue.OrderedExperiment(images, position=get_position)
        >>>

        Acces directe simple.
        >>> experiment[0]
        LaueDiagram(name='image_0')
        >>> experiment[1]
        LaueDiagram(name='image_1')
        >>> experiment[1:10:3]
        [LaueDiagram(name='image_1'), LaueDiagram(name='image_4'), LaueDiagram(name='image_7')]

        Acces directe organise.
        >>> experiment[0, 0, 0]
        LaueDiagram(name='image_0')
        >>> experiment[2, 3, 0]
        LaueDiagram(name='image_11')
        >>>

        Acces slice
        >>> experiment[:2, 1:3, 0]
        array([[[LaueDiagram(name='image_1')],
                [LaueDiagram(name='image_2')]],
        <BLANKLINE>
               [[LaueDiagram(name='image_5')],
                [LaueDiagram(name='image_6')]]], dtype=object)
        >>> experiment[0, 0, 2:5]
        array([[[LaueDiagram(name='image_24'), LaueDiagram(name='image_36'),
                 LaueDiagram(name='image_48')]]], dtype=object)
        >>> experiment[::-1, :2, -1]
        array([[[LaueDiagram(name='image_116')],
                [LaueDiagram(name='image_117')]],
        <BLANKLINE>
               [[LaueDiagram(name='image_112')],
                [LaueDiagram(name='image_113')]],
        <BLANKLINE>
               [[LaueDiagram(name='image_108')],
                [LaueDiagram(name='image_109')]]], dtype=object)
        >>>
        """
        if isinstance(item, (slice, int, np.integer)):
            return super().__getitem__(item)

        if not isinstance(item, tuple):
            raise ValueError("La clef doit etre de type, int, slice or tuple "
                f"not {type(item).__name__}.")
        if len(item) != 3:
            raise ValueError("Si l'element est un tuple, il doit contenir 3 "
                f" elements. x, y et t. Il en contient {len(item)}.")
        x, y, t = item
        size = np.prod(self.get_shape(), dtype=np.uint32)

        if all(isinstance(coord, (int, np.integer)) for coord in item):
            return super().__getitem__(self.get_index()[x, y] + t*size)

        x_index = np.arange(self.get_shape()[0])[x] if isinstance(x, slice) else [x]
        y_index = np.arange(self.get_shape()[1])[y] if isinstance(y, slice) else [y]
        if isinstance(t, slice): # Si il est pas nescessaire de lire tous les diagrames.
            if (    (t.start is None or t.start >= 0)
                and (t.stop is not None and t.stop > 0)
                and (t.step is None or t.step > 0)
                ) or ( # cas croissant ou cas decroissant
                    (t.start is not None and t.start > 0)
                and (t.stop is None or t.stop >= 0)
                and (t.step is not None and t.step < 0)):
                t_max = max(
                    (0 if t.stop is None else t.stop), # Cas croissant.
                    (0 if t.start is None else t.start)) # Cas decroissant.
            else:
                if len(self) == 0: # Si il faut lire tous les diagrames.
                    for _ in self:
                        pass
                t_max = (len(self) // np.prod(self.get_shape())) - 1
            t_index = np.arange(t_max+1)[t]
        else:
            t_index = [t]

        table = np.empty((len(x_index), len(y_index), len(t_index)), dtype=object)
        table[...] = [[[
                    self[x_, y_, t_]
                    for t_ in t_index
                ]
                for y_ in y_index
            ]
            for x_ in x_index
        ]
        return table
