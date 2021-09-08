#!/usr/bin/env python3

"""
Recherche les axes de zone sans aucune mise en forme.
C'est vraiment le coeur de la recherche atomisee.
"""

import numpy as np
try:
    import numexpr
except ImportError:
    numexpr = None
import psutil


def atomic_find_zone_axes(transformer, gnomonics, dmax, nbr, tol):
    """
    ** Fonction 'bas niveau' de recherche d'axes de zonnes. **

    Notes
    -----
    * Cette fonction n'est pas faite pour etre utilisee directement,
    il vaut mieux s'en servir a travers ``laue.experiment.base_experiment.Experiment.find_zone_axes``
    ou encore via ``laue.diagram.LaueDiagram.find_zone_axes`` car le context
    est mieu gere, les entrees sont plus simples et les sorties aussi.
    * Il n'y a pas de verifications sur les entrees car elles sont faite
    dans les methodes de plus haut niveau.
    * Cette fonction n'est pas parallelisee. Par contre la methode
    ``laue.experiment.base_experiment.Experiment.find_zone_axes`` gere nativement le parallelisme.
    * La seule raison d'utiliser cette fonction, c'est si le pic_search
    utilise n'est pas celui de ``laue.experiment.base_experiment.Experiment``. Sinon, l'utilisation
    de cette fonction ne fera qu'alourdir et ralentir votre code.

    Parameters
    ----------
    transformer : laue.core.geometry.transformer.Transformer
        Instance d'un objet capable de gerer formellement
        la transformee de hough. (Cet argument n'est pas present
        si on utilise les methodes ci dessus car il fait partie
        d'une ``laue.experiment.base_experiment.Experiment``.)
    gnomonics : np.ndarray
        Les positions des spots en coordonnees gnomonic.
        Il faut que ``x_gnomonic = gnomonic[0]``
        et que ``y_gnomonic = gnomonic[1]``.
    dmax
        Comme ``laue.diagram.LaueDiagram.find_zone_axes`` a la difference
        que ce parametre n'est pas factultatif.
    nbr
        Comme ``laue.diagram.LaueDiagram.find_zone_axes`` a la difference
        que ce parametre n'est pas factultatif.
    tol
        Comme ``laue.diagram.LaueDiagram.find_zone_axes`` a la difference
        que ce parametre doit etre fixe par vous et ne peut
        pas prendre tous seulle une valeur optimale.

    Returns
    -------
    angles : iterable
        Vecteur des angles des droites. C'est la premiere partie
        de la representation polaire des droites. (C'est l'angle
        algebrique entre l'axe x et un vecteur normal a la droite.)
    dists : iterable
        Vecteur des distances des droites. C'est la seconde partie
        de la representation polaire des droites. (C'est la plus courte
        distance entre l'origine et tous les points constituant la droite.)
    axes_spots_ind : list
        Vecteur des ensembles de spots lies a chaque droites.
        On a ``len(axes_spots_ind) == nbr_d_axe_de_zones``.
    spots_axes_ind : list
        Vecteur des indices des droites passant par chaque spot.
        On a ``len(spots_axes_ind) == nbr_de_spots``.

    Examples
    --------
    >>> import numpy as np
    >>> import laue
    >>> from laue.core.geometry.transformer import Transformer
    >>> transformer = Transformer()
    >>> gnomonics = np.array(
    ... [[ 3.13651353e-01,  3.09226930e-01,  2.94649661e-01,  3.01913261e-01,
    ...    2.65658647e-01,  2.53744185e-01,  2.59687364e-01,  2.14474797e-01,
    ...    2.05701679e-01,  9.09550861e-02,  6.84629381e-02,  1.66859716e-01,
    ...    1.60926506e-01,  8.80179554e-02,  7.13057593e-02,  8.63905624e-02,
    ...    7.30837137e-02, -1.65674724e-02, -4.08478454e-02,  1.19812461e-03,
    ...   -1.81540363e-02, -7.28605017e-02,  7.98366740e-02, -2.69038416e-03,
    ...   -9.78478342e-02,  8.00240133e-03,  2.74614431e-03, -5.23754954e-03,
    ...   -1.16145127e-02, -2.47761104e-02, -3.26450653e-02, -5.55472001e-02,
    ...   -6.60679415e-02, -7.90777430e-02, -9.15642828e-02, -1.12629071e-01,
    ...   -6.06376082e-02, -1.27878949e-01, -1.63820893e-01, -1.87639564e-01,
    ...   -1.83578789e-01, -8.52464810e-02, -2.52354264e-01, -2.09580392e-01,
    ...   -1.17581628e-01, -1.22668095e-01, -1.73766926e-01, -2.07243070e-01,
    ...   -1.95010900e-01, -2.18700320e-01, -2.02279896e-01, -2.58101851e-01,
    ...   -2.74817050e-01, -2.12902710e-01, -2.19868407e-01, -2.24622726e-01,
    ...   -2.59554148e-01, -2.73386180e-01, -3.15663189e-01, -2.61830509e-01,
    ...   -2.71107376e-01, -2.63712078e-01, -2.69344717e-01, -2.66537964e-01,
    ...   -3.35108876e-01, -3.09192955e-01, -3.52482527e-01, -3.19909692e-01,
    ...   -3.14402401e-01, -3.22377235e-01, -3.29087257e-01, -3.43805134e-01,
    ...   -3.71652663e-01, -3.84382367e-01, -4.15039361e-01, -4.21311647e-01,
    ...   -4.27436978e-01, -4.30567324e-01],
    ...  [-4.40919399e-01, -3.70396405e-01,  3.96707416e-01,  1.17593547e-02,
    ...   -2.98925638e-01,  3.22567523e-01,  1.10948607e-02, -2.15068594e-01,
    ...    2.35642120e-01, -5.35671413e-01,  5.54859519e-01, -1.36278614e-01,
    ...    1.54246926e-01, -4.00430471e-01,  4.17484373e-01, -3.19116771e-01,
    ...    3.35176021e-01, -5.54156780e-01,  5.68944812e-01, -4.50160027e-01,
    ...    4.64025259e-01, -5.64013302e-01,  7.02395430e-03,  3.92187923e-01,
    ...    5.76627076e-01, -1.22935735e-01,  1.33947819e-01, -1.47039399e-01,
    ...    1.57771528e-01, -1.82383612e-01,  1.92295000e-01, -2.38321751e-01,
    ...    2.47572735e-01, -2.80969173e-01,  2.89397061e-01, -3.41624111e-01,
    ...    4.03913576e-03,  3.49186361e-01, -4.34728622e-01, -4.77657378e-01,
    ...    4.41173941e-01,  3.43652675e-03, -5.94909608e-01,  4.83771175e-01,
    ...   -1.12376906e-01,  1.18172102e-01,  1.64722977e-03, -2.51506448e-01,
    ...   -1.59108326e-01,  2.54048705e-01,  1.61634743e-01, -3.57992381e-01,
    ...    3.59640747e-01,  8.34673643e-04, -1.03698038e-01,  1.05036855e-01,
    ...   -2.98487246e-01,  2.99389601e-01, -4.20235783e-01, -1.99290574e-01,
    ...    1.99483901e-01, -1.19844824e-01,  1.19439557e-01, -4.03501937e-04,
    ...   -3.66618216e-01, -2.27829859e-01,  3.64764214e-01,  2.26117536e-01,
    ...   -1.36989250e-03, -1.41532809e-01,  1.38674900e-01, -1.92326447e-03,
    ...   -2.65554100e-01,  2.61379480e-01, -3.75563949e-01, -1.27799526e-01,
    ...    1.20914638e-01, -1.02077320e-01]])
    >>> dmax = 0.01086181640625
    >>> nbr = 7
    >>> tol = 0.01758723266
    >>> angles, dists, axes_spots_ind, spots_axes_ind = laue.atomic_find_zone_axes(
    ...     transformer, gnomonics, dmax, nbr, tol)
    >>> len(angles), len(dists)
    (9, 9)
    >>> for spots in axes_spots_ind:
    ...     print(sorted(spots))
    ...
    [1, 4, 7, 11, 22, 26, 28, 30, 32, 34, 37, 40, 43]
    [2, 5, 8, 12, 22, 25, 27, 29, 31, 33, 35, 38, 39, 42]
    [3, 6, 22, 36, 41, 46, 53, 63, 68, 71]
    [9, 10, 13, 14, 15, 16, 22]
    [9, 31, 44, 46, 55, 60, 66]
    [10, 23, 32, 45, 46, 54, 59, 64]
    [17, 35, 48, 54, 63, 70, 73]
    [18, 37, 50, 55, 63, 69, 72, 74]
    [42, 51, 52, 56, 57, 59, 60, 61, 62, 63]
    >>> spots_axes_ind[0]
    set()
    >>> sorted(spots_axes_ind[22])
    [0, 1, 2, 3]
    >>>
    """
    from laue.core.geometry.transformer import ind2comb

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

def _jump_find_zone_axes(args):
    """
    ** Help for ``LaueDiagram.find_zone_axes``. **

    Etale les arguments de ``atomic_find_zone_axes`` et saute la fonction si besoin.
    """
    transformer, gnomonics, dmax, nbr, tol = args
    if transformer is None: # Si il ne faut pas refaire les calculs
        return {"dmax": dmax, "nbr": nbr, "tol": tol}
    return atomic_find_zone_axes(transformer, gnomonics, dmax, nbr, tol)
