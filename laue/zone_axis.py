#!/usr/bin/env python3

"""
** Represente un axe de zone. **
--------------------------------

Notes
-----
* Un axe de zone n'est rien d'autre qu'un ensemble de spots
alignes dans un plan gnomonic et qui s'inscrit dans un diagramme.
"""

import collections
import math
import numbers

import numpy as np
try:
    import numexpr
except ImportError:
    numexpr = None
try:
    import psutil # Pour acceder a la memoire disponible.
except ImportError:
    psutil = None


__pdoc__ = {"ZoneAxis.__contains__": True,
            "ZoneAxis.__hash__": True,
            "ZoneAxis.__iter__": True,
            "ZoneAxis.__len__": True}


def distance(axis1, axis2, *, weight=.5):
    r"""
    ** Calcule la distance entre plusieurs axes de zones. **

    Parameters
    ----------
    axis1 : laue.zone_axis.ZoneAxis, tuple, list, np.ndarray
        Un ou plusieurs axes. Cette fonction accepte une liste
        d'axes de facon a pouvoir faire tous les calculs d'un
        seul coup. Cela fait gagner grandement en efficacite.
        Un axe peut etre represente par un tuple a 2 elements, chaque
        element etant respectivement l'angle et la distance de la droite
        a l'origine.
    axis2 : laue.zone_axis.ZoneAxis, tuple, list, np.ndarray
        Un ou plusieurs axes.
        Les distances calculees seront les distances entre toutes
        les combinaisons possibles d'axes entre ``axis1`` et ``axis2``.
    weight : float
        * Importance de l'angle entre les droites plutot que
        leurs distances par rapport a l'origine.
        * 0.0 => Seule la distance compte:
        \[ dist = \lvert d1-d2 \rvert \]
        * 0.5 => L'angle et la distance comptent pareillement:
        \[ dist = \sqrt{\lambda^2.(\pi - \lvert \lvert \theta1-\theta2 \rvert - \pi \rvert )^2 + (1-\lambda)^2.(d1-d2)^2} \]
        * 1.0 => Seule l'angle compte:
        \[ dist = \pi - \lvert \lvert \theta1-\theta2 \rvert - \pi \rvert \]

    Returns
    -------
    float, np.ndarray
        * Les distances entre les droites:
            * Si axis1 et axis2 sont unique (pas un cluster)
                la distance renvoyee est un simple flottant, image
                de la distance separant ces 2 droites.
            * Si axis1 et axis2 sont tous 2 des clusters regroupants
                plusieurs axes, la matrice de distances entre les 2
                clusteur est renvoyee sous la forme d'une array numpy.
                ``dist(ax1, ax2) = mat[ax1, ax2]``
            * Si l'un des 2 arguments seulement est un cluster, alors
                le vecteur des distances qui separent chacuns des axes
                du cluster a l'axe isole est renvoye.

    Examples
    -------
    >>> import laue
    >>> from laue.zone_axis import distance
    >>> ax1, ax2 = (-2.719, 0.2432), (0.02063, 0.0799)
    >>> image = "laue/examples/ge_blanc.mccd"
    >>> axes = laue.Experiment(image,
    ...     config_file="laue/examples/ge_blanc.det"
    ...     )[0].find_zone_axes()[:3]
    >>>
    >>> round(distance(ax1, ax2, weight=0.0), 2)
    0.16
    >>> round(distance(ax1, ax2, weight=1.0), 2)
    2.74
    >>> round(distance(ax1, ax2), 2)
    1.37
    >>>
    >>> type(distance(ax1, axes))
    <class 'numpy.ndarray'>
    >>> distance(ax1, axes).shape
    (3,)
    >>> type(distance(axes, axes))
    <class 'numpy.ndarray'>
    >>> distance(axes, axes).shape
    (3, 3)
    >>>
    """
    assert isinstance(axis1, (ZoneAxis, np.ndarray, list, tuple)), \
        f"'axis1' has to be a ZoneAxis or axis container, not a {type(axis1).__name__}."
    if isinstance(axis1, (np.ndarray, list)):
        assert len(axis1), "La liste des axes doit contenir au moins 1 element."
        assert all(isinstance(axis, (ZoneAxis, tuple, np.ndarray)) for axis in axis1)
        assert all(len(axis) == 2 for axis in axis1 if not isinstance(axis, ZoneAxis))
    if isinstance(axis1, tuple):
        assert len(axis1) == 2, f"Il ne doit y avoir que 2 coordonnees, pas {len(axis1)}."
        assert all(isinstance(c, numbers.Number) for c in axis1)
    assert isinstance(axis2, (ZoneAxis, np.ndarray, list, tuple)), \
        f"'axis2' has to be a ZoneAxis or axis container, not a {type(axis2).__name__}."
    if isinstance(axis2, (np.ndarray, list)):
        assert len(axis2), "La liste des axes doit contenir au moins 1 element."
        assert all(isinstance(axis, (ZoneAxis, tuple, np.ndarray)) for axis in axis2)
        assert all(len(axis) == 2 for axis in axis2 if not isinstance(axis, ZoneAxis))
    if isinstance(axis2, tuple):
        assert len(axis2) == 2, f"Il ne doit y avoir que 2 coordonnees, pas {len(axis2)}."
        assert all(isinstance(c, numbers.Number) for c in axis2)
    assert isinstance(weight, numbers.Number), \
        f"'weight' has to be a number not a {type(weight).__name__}."
    assert 0 <= weight <= 1, f"'weight' must be in [0, 1], not {weight}."

    # Simplification du probleme.
    if isinstance(axis1, (ZoneAxis, tuple)) and isinstance(axis2, (ZoneAxis, tuple)):
        return distance(
            np.array([axis1]),
            np.array([axis2]),
            weight=weight)[0, 0]
    if isinstance(axis1, (ZoneAxis, tuple)):
        return distance(
            np.array([axis1]),
            axis2,
            weight=weight)[0, :]
    if isinstance(axis2, (ZoneAxis, tuple)):
        return distance(
            axis1,
            np.array([axis2]),
            weight=weight)[:, 0]

    # Calcul des distances
    meth = lambda axis: axis.get_polar_coords() if isinstance(axis, ZoneAxis) else axis
    angle1, dist1 = np.array([meth(axis) for axis in axis1], dtype=np.float32).transpose()
    angle2, dist2 = np.array([meth(axis) for axis in axis2], dtype=np.float32).transpose()
    angle1, angle2 = np.meshgrid(angle1, angle2, indexing="ij", copy=False)
    dist1, dist2 = np.meshgrid(dist1, dist2, indexing="ij", copy=False)
    pi = np.float32(np.pi)

    if weight == 0:
        return np.abs(dist2-dist1)
    if weight == 1:
        return pi - np.abs(np.abs(angle2-angle1) - pi)
    if numexpr is not None:
        return numexpr.evaluate("sqrt(weight**2*(pi - abs(abs(angle2-angle1) - pi))**2 "
                                "+ (1-weight)**2*(dist2-dist1)**2)")
    return np.sqrt(
        weight**2*(pi - np.abs(np.abs(angle2-angle1) - pi))**2
      + (1-weight)**2*(dist2-dist1)**2)


class ZoneAxis:
    """
    Un axe de zone seul.
    """
    def __init__(self, diagram, spots_ind, identifier, angle, dist):
        """
        Notes
        -----
        * L'utilisateur n'a pas a generer des objets issus de cette classe.
        Ils sont generes automatiquement par des instances de ``laue.diagram.LaueDiagram``.
        * Il n'y a pas de verifications faites sur les entrees car l'utilisateur
        ne doit pas toucher a l'initialisateur. La performance devient priorite.

        Parameters
        ----------
        diagram : laue.diagram.LaueDiagram
            C'est le diagramme pere dans lequel est contenu l'axe de zone.
            C'est aussi lui qui contient vraiment les spots.
        spots_ind : iterable
            C'est la totalite des indices des spots qui constituent cet axe de zone.
            Les 'vrais' instances de spots (de type ``laue.spot.Spot``) sont accessiblent
            via le parametre ``diagram``.
        identifier : int
            Identifiant unique de l'axe de zone. Cet identifiant est unique au
            sein d'un diagramme mais pas forcement au sein d'une experience.
        theta : float
            Angle entre l'axe ``x`` et le vecteur normal a cet axe passant par l'origine.
        dist : float
            Distance entre cette droite et l'origine.
        """
        self.diagram = diagram
        self.spots = collections.OrderedDict(
            (
                (int(ind), diagram[ind])
                for ind in sorted(spots_ind)
            ))
        self.identifier = identifier
        self.angle = angle
        self.dist = dist

    def dist_mean(self):
        """
        ** La moyenne des distances entre les points et l'axe. **

        Returns
        -------
        float
            La moyenne des distances. Voir ``laue.geometry.Transformer.dist_line``.

        Examples
        --------
        >>> import numpy as np
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = sorted(diag.find_zone_axes(), key=lambda a: len(a)-a.get_quality())[0]
        >>>
        >>> type(axis.dist_mean())
        <class 'numpy.float64'>
        >>>
        """
        theta_vect, dist_vect = self.get_polar_coords()
        theta_vect, dist_vect = np.array([theta_vect]), np.array([dist_vect])
        x_vect, y_vect = np.array([spot.get_gnomonic() for spot in self]).transpose()
        return self.diagram.experiment.transformer.dist_line(
            theta_vect, dist_vect, x_vect, y_vect).mean()

    def get_id(self):
        """
        ** Recupere le numero de cet axe. **

        Au seins d'un diagrame les axes ont des identifiants
        allant de [0, nbr_axes[.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> all(axes.get_id() == i for i, axes in enumerate(diag.find_zone_axes()))
        True
        >>>
        """
        return self.identifier

    def get_polar_coords(self):
        """
        ** Recupere l'angle et la distance representant l'axe. **

        Returns
        -------
        angle : float
            Le meme angle que celui donne a l'initialisateur.
        dist : float
            La meme distance que celle donnee a l'imitialisateur.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>>
        >>> angle, dist = axis.get_polar_coords()
        >>> type(angle) # In radian.
        <class 'numpy.float32'>
        >>> type(dist) # In mm.
        <class 'numpy.float32'>
        >>>
        """
        return self.angle, self.dist

    def get_quality(self):
        """
        ** Estime la qualite de cet axe de zone. **

        Returns
        -------
        float
            * Une grandeur qui permet d'estimet le nombre
                et la proximite des spots lies a cet axe.
                * 0.0 => Axes de zone tres mauvais.
                * 0.5 => Axe de zone moyen.
                * 1.0 => Axe de zone exeptionel.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> qualities = [axis.get_quality() for axis in diag.find_zone_axes()]
        >>>
        >>> 0 < min(qualities) <= max(qualities) < 1
        True
        >>>
        """
        def dmean_2_score(dmean, d_max=.0117):
            """
            f([0, d_min/4[) = [1, .9[
            f([d_min/4, d_max[) = [.9, .2[
            f([d_max, +oo[) = [.2, 0[
            """ 
            d_min = d_max / 4
            
            if dmean < d_min:
                return 1 - ((1-.9)/d_min)*dmean
            slope = (.9-.2)/(d_max-d_min)
            if dmean < d_max:
                return .9 - slope*(dmean-d_min)
            return .2*math.exp(-(slope/.2)*(dmean-d_max))

        def nbr_2_score(nbr, nbr_min=7):
            """
            f(nbr_min) = 0.1
            f(2*nbr_min) = 0.8
            f(+00) = 1
            """
            nbr_max = 2 * nbr_min
            a, b = .1, .8
            lna = math.log((1-a)/a)
            lnb = math.log((1-b)/b)
            mu = (nbr_min*lnb - nbr_max*lna) / (lnb - lna)
            lamb = lna / (mu - nbr_min)
            return 1 / (1 + math.exp(-lamb*(nbr-mu)))

        nbr_weight = .75 # Importance du nombre de points par raport a la proximite.
        return nbr_weight*nbr_2_score(len(self)) + (1-nbr_weight)*dmean_2_score(self.dist_mean())

    def plot_gnomonic(self, axe_pyplot=None, *, display=True):
        """
        ** Affiche un axe de zone dans le plan gnomonic. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte la methode ``.axline``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import laue
        >>>
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>>
        >>> axis.plot_gnomonic(display=False)
        <AxesSubplot:title={'center':'plan gnomonic'}, xlabel='x.Gi (mm)', ylabel='y.Gj (mm)'>
        >>>
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> axis.plot_gnomonic(axe, display=False)
        <AxesSubplot:>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()
            axe_pyplot.set_title("plan gnomonic")
            axe_pyplot.set_xlabel("x.Gi (mm)")
            axe_pyplot.set_ylabel("y.Gj (mm)")

        normal = np.array([math.cos(self.angle), math.sin(self.angle)])
        director = np.array([math.sin(self.angle), -math.cos(self.angle)])
        point1 = self.dist * normal
        point2 = point1 + director
        axe_pyplot.axline(point1, point2, lw=0.5, color="gray", clip_box=((-.1, -.1), (.1, .1)))

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def plot_xy(self, axe_pyplot=None, *, display=True):
        """
        ** Affiche un axe de zone tordu dans le plan de la camera. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte la methode ``.plot``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import laue
        >>>
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>>
        >>> axis.plot_xy(display=False)
        <AxesSubplot:title={'center':'plan camera'}, xlabel='x.Ci (pxl)', ylabel='y.Cj (pxl)'>
        >>>
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> axis.plot_xy(axe, display=False)
        <AxesSubplot:>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()
            axe_pyplot.set_title("plan camera")
            axe_pyplot.set_xlabel("x.Ci (pxl)")
            axe_pyplot.set_ylabel("y.Cj (pxl)")

        normal = np.array([math.cos(self.angle), math.sin(self.angle)])
        director = np.array([math.sin(self.angle), -math.cos(self.angle)])
        centre = self.dist * normal
        points_gnom_x = centre[0] + np.linspace(-2.0, 2.0, 50)*director[0]
        points_gnom_y = centre[1] + np.linspace(-2.0, 2.0, 50)*director[1]
        cam_x, cam_y = self.diagram.experiment.transformer.gnomonic_to_cam(
            points_gnom_x, points_gnom_y, self.diagram.experiment.set_calibration())
        cam_x_max, cam_y_max = self.diagram.experiment.get_images_shape()
        to_keep = (cam_x <= cam_x_max) & (cam_x >= 0) & (cam_y <= cam_y_max) & (cam_y >= 0)
        cam_x, cam_y = cam_x[to_keep], cam_y[to_keep]
        axe_pyplot.plot(cam_x, cam_y, color="blue")

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def __contains__(self, spot):
        """
        ** Verifie que le spot est dans l'axe de zone. **

        Parameters
        ----------
        spot : laue.spot.Spot, int
            L'instance de spot dont on cherche a savoir
            si il est present ou pas. Ou bien son indice
            au sein de ``laue.diagram.LaueDiagram``.

        Returns
        -------
        boolean
            True si le spot est lie a cet axe de zone, False sinon.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>> spot0, i_ok = diag[0], next(iter(axis.spots))
        >>> type(spot0), type(i_ok)
        (<class 'laue.spot.Spot'>, <class 'int'>)
        >>>
        >>> spot0 in axis
        False
        >>> i_ok in axis
        True
        >>> 0 in axis
        False
        >>> diag[i_ok] in axis
        True
        >>>
        """
        if isinstance(spot, int):
            return spot in self.spots

        from laue.spot import Spot
        if isinstance(spot, Spot):
            return spot in self.spots.values()

        raise AssertionError("'spot' has to be an instance of "
            f"Spot or int, not {type(spot).__name__}.")

    def __hash__(self):
        """
        ** Permet de faire des tables de hachage. **

        Returns
        -------
        int
            Identifiant "unique" (du moins le plus possible) representant cet axe de zone.
        """
        return hash((self.diagram, self.identifier))

    def __iter__(self):
        """
        ** Cede les spots de cet axe. **

        Yields
        ------
        spot : laue.spot.Spot
            Les spots de cet axe de zone par indice croissant.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>> for spot in axis:
        ...     pass
        ...
        >>>
        """
        yield from self.spots.values()

    def __len__(self):
        """
        ** Renvoie le nombre de points contenus dans cet axe de zone. **

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> axis = diag.find_zone_axes()[0]
        >>> type(len(axis))
        <class 'int'>
        >>> len(axis) >= 6
        True
        >>>
        """
        return len(self.spots)

    def __str__(self):
        """
        ** Renvoie une jolie representation de l'axe. **
        """
        return ("Axe de zone:\n"
                f"\tdiagram: {self.diagram.get_id()}\n"
                f"\tspots: {tuple(self.spots.keys())}\n")

    def __repr__(self):
        """
        ** Renvoie une chaine evaluable de self. **
        """
        return ("ZoneAxis("
                # f"diagram={repr(self.diagram)}, " # Affichage bien trop gourmand.
                f"spots_ind={tuple(self.spots.keys())}, "
                f"identifier={self.identifier}, "
                f"angle={self.angle:.4f}, "
                f"dist={self.dist:.4f})")


def _get_zone_axes_pickle(args):
    """
    ** Help for ``LaueDiagram.find_zone_axes``. **

    * Aucune verification n'est faite car l'utilisateur ne doit pas
    y mettre les pieds.

    Returns
    -------
    angles : iterable
        Vecteur des angles des droites.
    dists : iterable
        Vecteur des distances des droites.
    axes_spots_ind : iterable
        Vecteur des ensembles de spots lies a chaque droites.
    spots_axes_ind : iterable
        Vecteur des indices des droites passant par chaque spot.
    """
    transformer, gnomonics, dmax, nbr, tol = args

    from laue.geometry import ind2comb

    # Parsing des arguments
    if isinstance(transformer, bytes): # Si il est serialise.
        import cloudpickle
        transformer = cloudpickle.loads(transformer) # On le deserialise.

    # Recherches des axes de zone.
    angles, dists = transformer.hough_reduce(
        *transformer.hough(*gnomonics),
        nbr=nbr, tol=tol) # Recuperation des axes.
    if len(angles) <= 1: # Si on a pas trouve suffisement de choses.
        return (), (), (), ((),)*gnomonics.shape[-1]

    # Attribution des points aux droites.
    axes_spots_ind = [set() for _ in range(len(angles))] # A chaque droite, c'est les spots qu'elle possede.
    spots_axes_ind = [set() for _ in range(gnomonics.shape[-1])]
    x_inters, y_inters = transformer.inter_lines(angles, dists)
    xg_spots, yg_spots = gnomonics

    ## Recuperation des points aux intersections.
    
    ### Calcul des points les plus proche pour chaque intersections.
    used_memory = len(x_inters)*len(xg_spots)*8 # Taille de la matrice de distance en octet.
    if psutil is not None and psutil.virtual_memory().available > 2*used_memory:
        xg_spots_mesh, x_inters_mesh = np.meshgrid(xg_spots, x_inters, copy=False)
        yg_spots_mesh, y_inters_mesh = np.meshgrid(yg_spots, y_inters, copy=False)
        if numexpr is not None: # d[inter, gnomo]
            distances = numexpr.evaluate(
                "sqrt((xg_spots_mesh-x_inters_mesh)**2 + (yg_spots_mesh-y_inters_mesh)**2)")
        else:
            distances = np.sqrt((xg_spots_mesh-x_inters_mesh)**2 + (yg_spots_mesh-y_inters_mesh)**2)
        nearest_spots = np.argmin(distances, axis=1) # Pour chaque intersections, son spot le plus proche.
        del distances, xg_spots_mesh, x_inters_mesh, yg_spots_mesh, y_inters_mesh
    else: # Si il n'y a pas suffisement de RAM.
        nearest_spots = np.array([ # attention 'numexpr' est 7 fois plus lent.
            np.argmin(np.sqrt((xg_spots-x_inter)**2 + (yg_spots-y_inter)**2))
            for x_inter, y_inter in zip(x_inters, y_inters)], dtype=int)

    ### Selection des bons candidats.
    spots_left = [] # Les spots non references.
    for spot_ind, (xg_pic, yg_pic) in enumerate(zip(xg_spots, yg_spots)):
        inters_cand = np.argwhere(nearest_spots == spot_ind)
        adds_inter = inters_cand[
            (x_inters[inters_cand]-xg_pic)**2
          + (y_inters[inters_cand]-yg_pic)**2
          < dmax**2]
        if adds_inter.any():
            _adds_axes_1, _adds_axes_2 = ind2comb(adds_inter, n=len(angles))
            adds_axes = set(_adds_axes_1) | set(_adds_axes_2)
            for add_axis in adds_axes:
                axes_spots_ind[add_axis].add(spot_ind)
            spots_axes_ind[spot_ind].update(adds_axes) # f"le spot num {spot_ind} est l'intersections des axes {adds_axes}."
        else:
            spots_left.append(spot_ind)

    ## Recuperation des points colles a un seul axe.
    spots_left = np.array(spots_left, dtype=int) # Les indices des spots restants.

    used_memory = len(spots_left)*len(angles)*8 # Taille memoire de la matrice de distances.
    if psutil is not None and psutil.virtual_memory().available > 2*used_memory:
        distances = transformer.dist_line( # d[line, point]
            angles, dists, xg_spots[spots_left], yg_spots[spots_left])
        axis_ind = np.argmin(distances, axis=0) # A chaque points, indice de la droite la plus proche.
        close_spots = distances.min(axis=0) < dmax # La matrice des points suffisement proches.
        del distances
        for axis_ind, spot_left in zip(axis_ind[close_spots], spots_left[close_spots]):
            axes_spots_ind[axis_ind].add(spot_left)
            spots_axes_ind[spot_left].add(axis_ind)   
    else: # Si il n'y a pas suffisement de RAM.
        for spot_left, xg_pic, yg_pic in zip(spots_left, xg_spots[spots_left], yg_spots[spots_left]):
            xg_pic, yg_pic = np.array([xg_pic], dtype=np.float32), np.array([yg_pic], dtype=np.float32)
            distances = transformer.dist_line(angles, dists, xg_pic, yg_pic) # d[line, point]
            axis_ind = np.argmin(distances[:, 0])
            if distances[axis_ind] < dmax:
                axes_spots_ind[axis_ind].add(spot_left)
                spots_axes_ind[spot_left].add(axis_ind)

    # Suppression des axes qui contiennent pas suffisement de points.
    mask_axes_to_keep = np.array([len(spots_ind) for spots_ind in axes_spots_ind]) >= nbr
    ind_axes_to_keep = set(np.argwhere(mask_axes_to_keep)[:, 0])
    spots_axes_ind = [axes_ind & ind_axes_to_keep for axes_ind in spots_axes_ind]
    axes_spots_ind = [spots_ind for axis_ind, spots_ind
                     in enumerate(axes_spots_ind)
                     if axis_ind in ind_axes_to_keep]
    angles, dists = angles[mask_axes_to_keep], dists[mask_axes_to_keep]

    # Changement des anciens par les nouveaux indices de droites.
    old_to_new = {
        old_axis_ind: new_axis_ind
        for new_axis_ind, old_axis_ind
        in enumerate(sorted(set.union(*spots_axes_ind)))}
    spots_axes_ind = [
        {old_to_new[old_axis_ind] for old_axis_ind in old_axes_ind}
        for old_axes_ind in spots_axes_ind]

    return angles, dists, axes_spots_ind, spots_axes_ind
