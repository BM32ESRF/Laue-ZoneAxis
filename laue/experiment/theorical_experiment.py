#!/usr/bin/env python3

"""
** Manipule un lot de simule de diagrammes. **
---------------------------------------------

Fait la meme chose que ``laue.experiment.base_experiment.Experiment``
a la difference que les donnees sont des donnes simulee et non pas experimentales.
"""

import itertools

import numpy as np

from laue.diagram import LaueDiagram
from laue.spot import Spot
from laue.experiment.base_experiment import Experiment
from laue.tools.image import create_image


class TheoricalExperiment(Experiment):
    """
    ** Permet de travailler sur des donnees theoriques. **
    """
    def __init__(self, positions=(), miller_inds=None, intensities=None, **kwargs):
        """
        Parameters
        ----------
        positions : iterable
            Les positions des pics pour chacuns des diagrammes.
            Chaque element de ``positions`` contient donc les
            positions des spots du diagramme considere,
            d'abord selon x, puis selon y. Ces elements peuvent
            etre de type ``list``, ``tuple`` ou ``np.ndarray``.
        miller_inds : iterable
            La liste des 3 indices de miller (hkl) pour chaque diagramme.
            L'ordre des elements des ``miller_inds`` doit etre strictement
            le meme que celui de ``positions``. La valeur None indique que
            les indices ne sont pas connus.
        intensities : iterable
            La liste des intensites de chacuns des spots. Les
            valeurs fournis ici se retrouverons dans ``laue.spot.Spot.get_intensity``.
        **kwargs
            Ils sont transmis a ``laue.experiment.base_experiment.Experiment.__init__``.
        """
        assert hasattr(positions, "__iter__"), ("'positions' must to be iterable. "
            f"It can not be of type {type(positions).__name__}.")
        if miller_inds is None:
            miller_inds = itertools.cycle((None,))
        assert hasattr(miller_inds, "__iter__"), ("'miller_inds' must to be iterable. "
            f"It can not be of type {type(miller_inds).__name__}.")
        if intensities is None:
            intensities = itertools.cycle((None,))
        assert hasattr(intensities, "__iter__"), ("'intensities' must to be iterable. "
            f"It can not be of type {type(intensities).__name__}.")

        images = (
            create_image(
                diag.get_positions(),
                intensities=[spot.get_intensity() for spot in diag],
                shape=self.get_images_shape())
            for diag in self)

        Experiment.__init__(self, images=images, **kwargs)
        self.positions = positions
        self.miller_inds = miller_inds
        self.intensities = intensities

    def get_diagrams(self, *, tense_flow=False):
        """
        Fait la meme chose que ``laue.experiment.base_experiment.Experiment.get_diagrams``.
        Seulement les diagrammes ne sont pas crees a partir d'images
        mais a partir des donnees simulees.
        """
        from laue.tools.multi_core import prevent_generator_size

        def update_len(func):
            """
            Tient a jour la longueur de l'experience.
            """
            def decorate(*func_args, **func_kwargs):
                for i, element in enumerate(func(*func_args, **func_kwargs)):
                    yield element
                self._len = i + 1

            return decorate

        @update_len
        @prevent_generator_size(min_size=1)
        def _diagram_extractor(self):
            """
            Premiere vraie lecture. Cede les diagrammes.
            """
            spot_im = np.array([[0]], dtype=np.uint16)
            distortion = 1.0

            for diag_num, (position, miller_ind, intensity) in enumerate(
                    zip(self.positions, self.miller_inds, self.intensities)):
                # Verifications generales.
                if not isinstance(position, (np.ndarray, list, tuple)):
                    raise TypeError(f"Les positions des pics du {diag_num+1}eme diagramme "
                        f"ne sont de type {type(positions).__name__}. Or seul np.ndarray, list ou tuple sont admis.")
                position = np.array(list(position), dtype=np.float32)
                if position.ndim != 2:
                    raise ValueError(f"Les positions des pics du {diag_num+1}eme diagramme "
                        "sont incomprehensible. Il doit y avoir la liste des x et des y, ce "
                        f"qui fait un tableau a 2 dimensions. Or il y a {position.ndim} dimensions.")
                if position.shape[1] == 2 and position.shape[0] != 2:
                    position = position.transpose()
                if miller_ind is not None and not isinstance(miller_ind, (np.ndarray, list, tuple)):
                    raise TypeError(f"Les indices de miller des pics du {diag_num+1}eme diagramme "
                        f"ne sont de type {type(miller_ind).__name__}. Or seul np.ndarray, list ou tuple sont admis.")
                if miller_ind is not None and len(miller_ind) != position.shape[1]:
                    raise ValueError(f"Les positions du {diag_num+1}eme diagramme laissent sous-entendre "
                        f"qu'il y a {position.shape[1]} spots. Or {len(miller_ind)} triplet d'indices de miller sont fournis.")
                if intensity is not None and not isinstance(intensity, (np.ndarray, list, tuple)):
                    raise TypeError(f"Les intensites des pics du {diag_num+1}eme diagramme "
                        f"ne sont de type {type(intensity).__name__}. Or seul np.ndarray, list ou tuple sont admis.")
                if intensity is not None and len(intensity) != position.shape[1]:
                    raise ValueError(f"Les positions du {diag_num+1}eme diagramme laissent sous-entendre "
                        f"qu'il y a {position.shape[1]} spots. Or {len(intensity)} intensitees sont fournies.")

                miller_ind = miller_ind if miller_ind is not None else itertools.cycle((None,))
                intensity = intensity if intensity is not None else itertools.cycle((1.0,))
                name = f"diagram_{diag_num}"
                laue_diagram = LaueDiagram(name, spots=[], experiment=self)

                for i, (pos_x, pos_y, hkl, inten) in enumerate(zip(*position, miller_ind, intensity)):
                    # Verifications pour les pics.
                    if hkl is not None and not isinstance(hkl, (np.ndarray, list, tuple)):
                        raise TypeError(f"Les indices hkl du {i+1}eme spot du {diag_num+1}eme diagramme "
                            f"sont de type {type(hkl).__name__}. Or seul np.ndarray, list ou tuple sont admis.")
                    hkl = np.array(hkl, dtype=int)
                    if hkl is not None and hkl.shape != (3,):
                        raise ValueError(f"Les indices hkl du {i+1}eme spot du {diag_num+1}eme diagramme "
                            f"ne sont pas 3, ils sont {hkl.shape}. C'est difficile a separer en 3 entiers!")
                    if not isinstance(inten, (int, float)):
                        raise ValueError(f"Les intensitee doivent etre des nombre, pas des {type(inten).__name__}.")
                    
                    hkl = tuple(hkl) if hkl is not None else hkl
                    bbox = (pos_x, pos_y, 1, 1)
                    laue_diagram.spots.append(Spot(
                        bbox=bbox, spot_im=spot_im, distortion=distortion, diagram=laue_diagram
                        ))
                    laue_diagram.spots[-1].intensity = inten
                    laue_diagram.spots[-1].position = (pos_x, pos_y)
                    laue_diagram.spots[-1].hkl = hkl

                yield laue_diagram

        if self._diagrams_iterator is None:
            self._diagrams_iterator = iter(_diagram_extractor(self))

        from laue.tools.multi_core import RecallingIterator
        return (
            (lambda x: (yield from x))(RecallingIterator(self._diagrams_iterator, mother=self))
            if tense_flow else set(RecallingIterator(self._diagrams_iterator, mother=self)))

    def get_images_shape(self):
        """
        ** Recupere les dimensions des fausses images. **

        Returns
        -------
        tuple
            (2048, 2048) valeur par defaut pour les donnes simulees.
        """
        return (2048, 2048)
