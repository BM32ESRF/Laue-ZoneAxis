#!/usr/bin/env python3

"""
** Interface d'aide a pickle. **
--------------------------------

Fournis des interfaces aux objets de base constituants
une experience de sorte a ce que pickle puisse
les serialiser rapidement et efficacement, en selectionant seulement
l'information utile.
"""

import collections
import os


__pdoc__ = {"SpotPickelable.__getstate__": True,
            "SpotPickelable.__setstate__": True,
            "ZoneAxisPickleable.__getstate__": True,
            "ZoneAxisPickleable.__setstate__": True,
            "DiagramPickleable.__getstate__": True,
            "DiagramPickleable.__setstate__": True,
            "TransformerPickleable.__getstate__": True,
            "TransformerPickleable.__setstate__": True,
            "ExperimentPickleable.__getstate__": True,
            "ExperimentPickleable.__setstate__": True}


class SpotPickleable:
    """
    ** Interface pour serialiser les spots. **
    """
    def __getstate__(self):
        """
        ** Extrait le contenu d'un spot. **

        Enregistre l'etat courant, ne fait pas de zele.
        """
        state = {}
        state["bbox"] = self.get_bbox()
        state["im"] = self.get_image()
        state["dis"] = self.get_distortion()
        state["id"] = self.get_id()
        return state

    def __setstate__(self, state):
        """
        ** Initialisateur pour pickle. **

        Examples
        --------
        >>> import pickle
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> pickle.loads(pickle.dumps(spot))
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>>
        """
        self.__init__(
            bbox=state["bbox"],
            spot_im=state["im"],
            distortion=state["dis"],
            diagram=None,
            identifier=state["id"])

class ZoneAxisPickleable:
    """
    ** Interface pour serialiser les axes. **
    """
    def __getstate__(self):
        """
        ** Recupere les informations pour reconstruir un axe. **

        Enregistre l'etat courant, ne fait pas de zele.
        """
        state = (
            tuple(self.spots.keys()),
            self.get_id(),
            *self.get_polar_coords())
        return state

    def __setstate__(self, state):
        """
        ** Initialisateur pour pickle. **

        Examples
        --------
        >>> import pickle
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = sorted(diag.find_zone_axes(), key=lambda a: len(a)-a.get_quality())[0]
        >>> axis
        ZoneAxis(spots_ind=(9, 31, 44, 46, 55, 60, 66), identifier=4, phi=-2.6836, mu=0.1541)
        >>> pickle.loads(pickle.dumps(axis))
        ZoneAxis(spots_ind=(9, 31, 44, 46, 55, 60, 66), identifier=4, phi=-2.6836, mu=0.1541)
        >>>
        """
        self.diagram = None
        self.spots = collections.OrderedDict(
            ((ind, None) for ind in state[0]))
        self._identifier = state[1]
        self._phi = state[2]
        self._mu = state[3]

class DiagramPickleable:
    """
    ** Interface pour serialiser un diagrame. **
    """
    def __getstate__(self):
        """
        ** Recupere les informations caracteristiques d'un diagrame. **

        Enregistre l'etat courant, ne fait pas de zele.
        """
        state = {}
        state["name"] = self.get_id()
        if not os.path.exists(self.get_id()):
            state["image"] = self.get_image_xy()

        state["spots"] = [spot.__getstate__() for spot in self]
        if self._axes:
            state["axes"] = {key: [axis.__getstate__() for axis in axes]
                for key, axes in self._axes.items()}
        if self._hkl:
            state["hkl"] = self._hkl
        return state

    def __setstate__(self, state):
        """
        ** Initialisateur pour pickle. **

        Examples
        --------
        >>> import pickle
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> diag
        LaueDiagram(name='laue/examples/ge_blanc.mccd')
        >>> pickle.loads(pickle.dumps(diag))
        LaueDiagram(name='laue/examples/ge_blanc.mccd')
        >>>
        """
        from laue.spot import Spot as Spot_
        from laue.zone_axis import ZoneAxis as ZoneAxis_

        class Spot(Spot_):
            def __init__(self, *args, state=None, **kwargs):
                if state is None:
                    super().__init__(*args, **kwargs)
                else:
                    self.__setstate__(state)

        class ZoneAxis(ZoneAxis_):
            def __init__(self, *args, state=None, **kwargs):
                if state is None:
                    super().__init__(*args, **kwargs)
                else:
                    self.__setstate__(state)

        self.__init__(state["name"], None)
        self._set_image(state.get("image", None))

        self._set_spots([Spot(state=s) for s in state["spots"]])
        for spot in self:
            spot.diagram = self
        if "axes" in state:
            self._axes = {key: [ZoneAxis(state=s) for s in axes]
                for key, axes in state["axes"].items()}
        for axes in self._axes.values():
            for axis in axes:
                axis.diagram = self
                axis.spots = collections.OrderedDict(
                    ((ind, self[ind]) for ind in axis.spots.keys()))

        self._hkl = state.get("hkl", {})

class TransformerPickleable:
    """
    ** Interface pour serialiser le gestionaire de transformation geometriques. **
    """
    def __getstate__(self):
        """
        ** Recupere les fonction vectorisee et symplifiees. **
        """
        state = {}
        state["verbose"] = self.verbose
        if self._fcts_cam_to_gnomonic:
            state["c2g"] = {k: v for k, v in self._fcts_cam_to_gnomonic.items()}
        if self._fcts_gnomonic_to_cam:
            state["g2c"] = {k: v for k, v in self._fcts_gnomonic_to_cam.items()}
        if self._fcts_cam_to_thetachi:
            state["c2t"] = {k: v for k, v in self._fcts_cam_to_thetachi.items()}
        if self._fcts_thetachi_to_cam:
            state["t2c"] = {k: v for k, v in self._fcts_thetachi_to_cam.items()}
        if self._parameters_memory:
            state["mem"] = self._parameters_memory
        return state

    def __setstate__(self, state):
        """
        ** Initialisateur pour pickle. **

        Examples
        --------
        >>> import pickle
        >>> from laue.core.geometry.transformer import Transformer
        >>> trans = pickle.loads(pickle.dumps(Transformer()))
        >>>
        """
        self.__init__(state["verbose"])
        if "c2g" in state:
            for k, v in state["c2g"].items():
                self._fcts_cam_to_gnomonic[k] = v
        if "g2c" in state:
            for k, v in state["g2c"].items():
                self._fcts_gnomonic_to_cam[k] = v
        if "c2t" in state:
            for k, v in state["c2t"].items():
                self._fcts_cam_to_thetachi[k] = v
        if "t2c" in state:
            for k, v in state["t2c"].items():
                self._fcts_thetachi_to_cam[k] = v
        if "mem" in state:
            self._parameters_memory = state["mem"]

class ExperimentPickleable:
    """
    ** Interface pour serialiser une experience. **
    """
    def __getstate__(self):
        """
        ** Recupere des informations d'une experience. **
        """
        state = {}

        state["verbose"] = self.verbose
        state["max_space"] = self.max_space
        state["threshold"] = self.threshold
        state["font_size"] = self.font_size
        state["ignore_errors"] = self.ignore_errors
        state["kwargs"] = self.kwargs
        state["kernel_font"] = self.kernel_font
        state["threshold"] = self.threshold
        state["kernel_dilate"] = self.kernel_dilate
        state["threshold"] = self.threshold
        state["mean_bg"] = self._mean_bg
        state["shape"] = self._shape
        state["mean_bg"] = self._mean_bg
        state["calibration_parameters"] = self._calibration_parameters
        state["gnomonic_matrix"] = self._gnomonic_matrix
        state["saving_file"] = self.saving_file
        state["compress"] = self.compress
        state["dt"] = self.dt

        state["images"] = None
        state["transformer"] = None
        state["predictors"] = None
        state["images_iterator"] = None
        state["diagrams_iterator"] = None
        state["axes_iterator"] = None
        state["subsets_iterator"] = None

        return state

    def __setstate__(self, state):
        """
        ** Initialise partiellement l'experience. **

        A partir des informations presentes dans ``state``,
        certains attributs sont crees ou completes.

        Examples
        --------
        >>> import pickle
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> laue.Experiment(image)
        Experiment('laue/examples')
        >>> pickle.loads(pickle.dumps(_))
        Experiment('laue/examples')
        >>>
        """
        if not hasattr(self, "verbose"):
            self.verbose = state["verbose"]
        self.max_space = state["max_space"]
        self.threshold = state["threshold"]
        self.font_size = state["font_size"]
        if not hasattr(self, "ignore_errors"):
            self.ignore_errors = state["ignore_errors"]
        self.kwargs = state["kwargs"]
        self.kernel_font = state["kernel_font"]
        self.kernel_dilate = state["kernel_dilate"]
        self._mean_bg = state["mean_bg"]
        self._shape = state["shape"]
        self._calibration_parameters = state["calibration_parameters"]
        self._gnomonic_matrix = state["gnomonic_matrix"]
        if not hasattr(self, "saving_file"):
            self.saving_file = state["saving_file"]
        if not hasattr(self, "compress"):
            self.compress = state["compress"]
        if not hasattr(self, "dt"):
            self.dt = state["dt"]
