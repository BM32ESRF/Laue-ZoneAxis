#!/usr/bin/env python3

"""
** Represente un spot dans un diagramme de Laue. **
---------------------------------------------------

L'avantage d'avoir un objet dedie aux spots, c'est que ca permet
d'avoir un code vraiment plus clair, ce qui est privilegie a une performance
legerement amoindrie.
"""

import math

import numpy as np


__pdoc__ = {"Spot.__hash__": True,
            "Spot.__sub__": True}


def distance(spot1, spot2, *, space="camera"):
    """
    ** Calcul la distance entre plusieur spots. **

    C'est une generalisation de la methode ``Spot.__sub__``.

    Parameters
    ----------
    spot1 : Spot, tuple, list, np.ndarray
        Un spot ou une liste de spots. Cette fonction accepte une liste
        de spots pour pouvoir faire tous les calculs d'un seul coup et gagner
        grandement en temps de calcul.
        Un spot peu etre representer par un tuple a 2 elements, chaque element
        etant la premiere et la seconde coordonnees du spot dans l'espace considere.
    spot2 : Spot, tuple, list, np.ndarray
        Un spot ou une autre liste de spots. Les distances calculees seront les distances
        entre toutes les combinaison possible de spots entre ``spot1`` et ``spot2``.
        Un spot peu etre representer par un tuple a 2 elements, chaque element
        etant la premiere et la seconde coordonnees du spot dans l'espace considere.
    space : str
        * "camera" => Distance euclidienne (en pixel) dans le plan de la camera.
        * "gnomonic" => Distance euclidienne (en mm) dans le plan gnomonic.
        * "angle" => Cosine distance (en degre) entre les vecteurs ``uq`` (axe de reflexion).

    Returns
    -------
    float, np.ndarray
        - Les distances entre les spots:
            - Si spot1 et spot2 sont des spots unique et non pas des listes,
            un flotant est renvoye, image de la distance entre ces 2 spots.
            - Si spot1 et spot2 sont tous deux des listes, retourne la matrice de
            distance entre les spots. ``dist(p1, p2) == mat[p1, p2]``
            - Si l'un des 2 spots est une liste et l'autre un spot, retourne
            la liste des distance qui separe ce spot unique a tous les autres.

    Examples
    -------
    >>> import laue
    >>> from laue.spot import distance
    >>> image = "laue/examples/ge_blanc.mccd"
    >>> diag = laue.Experiment(image)[0]
    >>> spot1, spot2 = diag[:2]
    >>> spot1.get_position(), spot2.get_position()
    ((1370.5171990171991, 1874.7800982800984), (1303.6322254335262, 1808.7420520231212))
    >>>
    >>> distance(spot1, spot2)
    93.99273
    >>> distance(spot1, diag[:5])
    array([  0.     ,  93.99273, 811.10895, 484.83295, 248.53378],
          dtype=float32)
    >>> distance(diag[:5], diag[:10]).shape
    (5, 10)
    >>>
    """
    assert isinstance(spot1, (Spot, np.ndarray, list, tuple)), \
        f"'spot1' has to be a Spot or list, not a {type(spot1).__name__}."
    if isinstance(spot1, (np.ndarray, list)):
        assert len(spot1), "La liste des spots doit contenir au moins 1 element."
        assert all(isinstance(spot, (Spot, tuple, np.ndarray)) for spot in spot1)
        assert all(len(spot) == 2 for spot in spot1 if hasattr(spot, "__iter__"))
    if isinstance(spot1, tuple):
        assert len(spot1) == 2, f"Il ne doit y avoir que 2 coordonnees, pas {len(spot1)}."
        assert all(isinstance(c, (float, int)) for c in spot1)
    assert isinstance(spot2, (Spot, np.ndarray, list, tuple)), \
        f"'spot2' has to be a Spot or list, not a {type(spot2).__name__}."
    if isinstance(spot2, (np.ndarray, list)):
        assert len(spot2), "La liste des spots doit contenir au moins 1 element."
        assert all(isinstance(spot, (Spot, tuple, np.ndarray)) for spot in spot2)
        assert all(len(spot) == 2 for spot in spot2 if hasattr(spot, "__iter__"))
    if isinstance(spot2, tuple):
        assert len(spot2) == 2, f"Il ne doit y avoir que 2 coordonnees, pas {len(spot2)}."
        assert all(isinstance(c, (float, int)) for c in spot2)
    assert space in {"camera", "gnomonic", "angle"}, f"'space' can not be {repr(space)}."

    # Simplification du probleme.
    if isinstance(spot1, (Spot, tuple)) and isinstance(spot2, (Spot, tuple)):
        return distance(
            np.array([spot1]),
            np.array([spot2]),
            space=space)[0, 0]
    if isinstance(spot1, (Spot, tuple)):
        return distance(
            np.array([spot1]),
            spot2,
            space=space)[0, :]
    if isinstance(spot2, (Spot, tuple)):
        return distance(
            spot1,
            np.array([spot2]),
            space=space)[:, 0]

    try:
        import numexpr
    except ImportError:
        numexpr = None

    # Cas ou l'on doit calculer une matrice de distances.
    if space == "angle":
        raise NotImplementedError

    if space == "camera":
        meth = lambda spot: spot.get_position() if isinstance(spot, Spot) else spot
    elif space == "gnomonic":
        meth = lambda spot: spot.get_gnomonic() if isinstance(spot, Spot) else spot
    x1, y1 = np.array([meth(spot) for spot in spot1], dtype=np.float32).transpose()
    x2, y2 = np.array([meth(spot) for spot in spot2], dtype=np.float32).transpose()
    x1, x2 = np.meshgrid(x1, x2, indexing="ij", copy=False)
    y1, y2 = np.meshgrid(y1, y2, indexing="ij", copy=False)

    if numexpr is not None:
        return numexpr.evaluate("sqrt((x1-x2)**2 + (y1-y2)**2)")
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


class Spot:
    """
    Represente un spot sur un diagramme de laue.
    """
    def __init__(self, bbox, spot_im, distortion, diagram):
        """
        ** Initialisation du spot. **

        Notes
        -----
        * L'utilisateur n'a pas a generer des objets issus de cette classe.
        Ils sont generes automatiquement par des instances de
        ``laue.experiment.base_experiment.Experiment`` et ``laue.diagram.LaueDiagram``.
        * Il n'y a pas de verifications faites sur les entrees car l'utilisateur
        ne doit pas toucher a l'initialisateur. La performance passe donc avant
        l'enorme mefiance envers les humains.

        Parameters
        ----------
        bbox : tuple
            Bounding Boxe (x, y, w, h) du spot dans l'image.
        spot_im : np.ndarray(np.uint16)
            Le bout de l'image en niveau de gris qui represente le spot.
            Le fond diffus doit etre deja enleve de l'image.
        distortion : float
            Le facteur de distortion accessible par l'accesseur ``Spot.get_distortion``.
        diagram : LaueDiagram
            Le diagram qui contient ces spots. De sorte a pouvoir remonter.
        """
        # Constantes.
        self.x, self.y, self.w, self.h = bbox
        self.spot_im = spot_im
        self.distortion = distortion # Facteur de diformite.
        self.diagram = diagram # Le contenaur.

        # Declaration des variables futur.
        self.intensity = None # Intensite du spot.
        self.position = None # Coordonnees x, y du baricentre dans le plan de la camera.
        self.gnomonic = None # Coordonnees x, y du baricentre projete dans le plan gnomonic.
        self.twicetheta_chi = None # Angles du rayon diffractes ayant engendre ce point.
        self.quality = None # Beautee du point.
        self.hkl = None # Les 3 indices de Miller h, k et l.
        self.axes = None

    def get_bbox(self):
        """
        ** Retourne les coordonnees de la boite. **

        Les coordonnees sont exprimes en pxl dans le plan de la camera.

        Returns
        -------
        tuple
            x, y, w, h

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> spot.get_bbox()
        (1368, 1873, 6, 5)
        >>>
        """
        return self.x, self.y, self.w, self.h

    def get_distortion(self):
        r"""
        ** Scalaire qui caracterise la rondeur de la tache. **

        \[ distortion = \frac{2 . \sqrt{\pi . area}}{girth} \]

        Returns
        -------
        float
            1.0 => Tache bien ronde
            0.0 => Tache biscornue.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> round(spot.get_distortion(), 4)
        0.8472
        >>>
        """
        return self.distortion

    def get_gnomonic(self):
        """
        ** Cherche les coordonnees dans le plan gnomonic **

        Returns
        -------
        tuple
            Les coordonnees x, y de la tache dans le plan gnomonic (en mm). de type (float, float)

        Examples
        --------
        >>> import laue
        >>> import numpy as np
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> type(spot.get_gnomonic())
        <class 'tuple'>
        >>> np.round(spot.get_gnomonic(), 4)
        array([ 0.3137, -0.4409])
        >>>
        """
        if self.gnomonic is not None:
            return self.gnomonic
        detector_parameters = self.diagram.experiment.set_calibration()
        xg, yg = self.diagram.experiment.transformer.cam_to_gnomonic(
            *self.get_position(), detector_parameters)
        self.gnomonic = (xg, yg)
        return self.gnomonic

    def get_hkl(self):
        """
        ** Recupere les indices de Miller. **

        Returns
        -------
        tuple
            Les 3 indices de Miller h, k et l dans un tuple (int, int, int).
        """
        raise NotImplementedError("Demandez a Ravis, il sera ravis de vous repondre!")

    def get_intensity(self):
        r"""
        ** Calcul l'intensite de la tache. **

        \[ intensity = \sum_{i \in bbox} pxl[i] - background[i] \]

        Returns
        -------
        int
            La somme des pixels qui constituent la tache.
            Seules les valeurs au dessus du fond diffus sont considerees.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> spot.get_intensity()
        814
        >>>
        """
        if self.intensity is not None:
            return self.intensity
        self.intensity = self.spot_im.sum()
        return self.intensity

    def get_position(self):
        r"""
        ** Calcul le centre d'inertie de la tache. **

        \[ position = \sum_{i, j \in bbox} (i, j) . \frac{pxl[i, j] - background[i, j]}{intensity} \]

        Returns
        -------
        tuple
            Les coordonnees x, y du centre de gravite de la tache (en pxl). de type (float, float).

        Examples
        --------
        >>> import numpy as np
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> np.round(spot.get_position(), 4)
        array([1370.5172, 1874.7801])
        >>>
        """
        if self.position is not None:
            return self.position
        x, y = (self.spot_im/self.get_intensity() * np.array(np.meshgrid(
                np.arange(self.x, self.x+self.w), np.arange(self.y, self.y+self.h)))
                ).reshape((2, -1)).sum(axis=1)
        self.position = (x, y)
        return x, y

    def get_quality(self):
        """
        ** Estime la qualite du point. **

        Returns
        -------
        float
            * Une grandeur qui permet d'estimer l'intensite et la rondeur du spot.
                - 0.0 => Tres moche, peu intense et tout disordu.
                - 1.0 => Tres joli, intense et rond.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image)[0][0]
        >>> spot
        Spot(position=(1370.52, 1874.78), quality=0.573)
        >>> round(spot.get_quality(), 2)
        0.57
        >>>
        """
        if self.quality is not None:
            return self.quality

        cout_ref = 100_000
        val_cout_ref = 0.95
        distortion_weight = 0.667

        a = -math.log(1-val_cout_ref) / cout_ref
        self.quality = (1-distortion_weight)*(1 - math.exp(-a*self.get_intensity())) + distortion_weight*self.get_distortion()
        return self.quality

    def get_theta_chi(self):
        """
        ** Cherche les angles du rayon reflechi. **

        Returns
        -------
        twicetheta : float
            L'angle entre l'axe ``x`` (ie axe du rayon incident) et la
            projection sur le plan ``(x, y)`` de l'axe du rayon reflechit.
        chi : float
            L'angle entre l'axe ``y`` et la projection sur le plan
            ``(y, z)`` de l'axe du rayon reflechit.
        """
        raise NotImplementedError

    def find_zone_axes(self):
        """
        ** Renvoi les axes de zone qui contienent ce point. **

        returns
        -------
        set
            L'ensemble des axes de zone de type ``laue.zone_axis.ZoneAxis``
            tel que ce spot appartient a ces axes. Si ce spot
            est lie a aucun axe, un ensemble vide est renvoye.
            Si ce spot est a l'intersection de plusieurs axes,
            l'ensemble contiendra plusieurs elements.

        Examples
        -------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> spot = diag.select_spots(n=1, sort="quality").pop()
        >>> type(spot.find_zone_axes())
        <class 'set'>
        >>> type(spot.find_zone_axes().pop())
        <class 'laue.zone_axis.ZoneAxis'>
        >>>
        """
        if self.axes is not None:
            return self.axes
        self.diagram.find_zone_axes() # met automatiquement a jour self.axes.
        return self.axes

    def plot_gnomonic(self, axe_pyplot=None, *, display=True):
        """
        ** Affiche ce spot dans le plan gnomonic. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte la methode ``.scatter``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0][0]
        >>>
        >>> spot.plot_gnomonic(display=False)
        <AxesSubplot:title={'center':'plan gnomonic'}, xlabel='x.Gi (mm)', ylabel='y.Gj (mm)'>
        >>>
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> spot.plot_gnomonic(axe, display=False)
        <AxesSubplot:>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()
            axe_pyplot.set_title("plan gnomonic")
            axe_pyplot.set_xlabel("x.Gi (mm)")
            axe_pyplot.set_ylabel("y.Gj (mm)")

        axe_pyplot.scatter(*self.get_gnomonic(), color="black")

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def plot_xy(self, axe_pyplot=None, *, display=True):
        """
        ** Affiche ce spot dans le plan de la camera. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte les methodes ``.plot`` et ``.scatter``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> spot = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0][0]
        >>>
        >>> spot.plot_xy(display=False)
        <AxesSubplot:title={'center':'plan camera'}, xlabel='x.Ci (pxl)', ylabel='y.Cj (pxl)'>
        >>>
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> spot.plot_xy(axe, display=False)
        <AxesSubplot:>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()
            axe_pyplot.set_title("plan camera")
            axe_pyplot.set_xlabel("x.Ci (pxl)")
            axe_pyplot.set_ylabel("y.Cj (pxl)")

        x, y, w, h = self.get_bbox()
        axe_pyplot.plot(
            [x-.5, x+w-.5, x+w-.5, x-.5, x-.5],
            [y-.5, y-.5, y+h-.5, y+h-.5, y-.5],
            color="grey")
        axe_pyplot.scatter(*self.get_position(), color="black")

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def _clean(self):
        """
        ** Vide les attributs recaculables. **

        Cela permet de rafraichir la valeur des attributs qui dependent
        d'une grandeur exterieur qui aurait changee. (Comme par example
        les parametres de set_calibration.)
        """
        self.gnomonic = None # Si jamais la set_calibration change.
        self.twicetheta_chi = None # Si jamais la set_calibration change.
        self.intensity = None # Si jamais l'image change.
        self.position = None # Si jamais l'image change.
        self.quality = None # Car on vient de changer 'self.intensity'.
        self.axes = None # Car si la projection gnomonique change, tout change.

    def __hash__(self):
        """
        ** Permet de faire des tables de hachage. **

        Returns
        -------
        int
            Identifiant "unique" (du moins le plus possible) representant ce spot.
        """
        return hash((*self.get_position(), self.get_intensity()))

    def __repr__(self):
        """
        ** Renvoie un representation evaluable de self. **
        """
        return (f"Spot("
                f"position=({self.get_position()[0]:.2f}, {self.get_position()[1]:.2f}), "
                f"quality={self.get_quality():.3f})")

    def __str__(self):
        """
        ** Offre une jolie representation. **
        """
        x, y = self.get_position()
        return ("Spot:\n"
                f"\tbbox: {self.get_bbox()}\n"
                f"\tposition: x={x}, y={y}\n"
                f"\tintensity: {self.get_intensity()}\n"
                f"\tdistortion: {self.get_distortion()}\n"
                f"\tquality: {self.get_quality()}\n")

    def __sub__(self, other):
        """
        ** Calcul la distance entre 2 taches. **

        Notes
        -----
        * La metrique utilisee est la metrique euclidienne
        dans le plan de la camera. Voir ``distance``.
        * Permet par example de construire facilement une matrice
        des distances avec ``np.meshgrid``.

        Returns
        -------
        float
            La distance en mm.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> spot1, spot2 = diag[:2]
        >>> round(spot1 - spot2)
        94
        >>>
        """
        return distance(self, other, space="camera")
