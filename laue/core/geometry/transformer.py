#!/usr/bin/env python3

"""
** Permet d'appliquer des transformations geometriques. **
----------------------------------------------------------

Notes
-----
Le module ``numexpr`` permet d'accelerer les calculs si il est installe.
"""

import collections
import math
import multiprocessing
import numbers
import os

import cloudpickle
import numpy as np
try:
    import numexpr
except ImportError:
    numexpr = None

from laue.core.geometry.symbolic import Compilator


__all__ = ["Transformer", "comb2ind", "ind2comb"]


class Transformer(Compilator):
    """
    Permet d'effectuer des transformations geometrique comme jongler
    entre l'espace de la camera et l'espace gnomonique ou encore
    s'ammuser avec la transformee de Hough.
    """
    def __init__(self):
        Compilator.__init__(self) # Globalisation des expressions.

        # Les memoires tampon.
        self._fcts_cam_to_gnomonic = collections.defaultdict(lambda: 0) # Fonctions vectorisees avec seulement f(x_cam, y_cam), les parametres sont deja remplaces.
        self._fcts_gnomonic_to_cam = collections.defaultdict(lambda: 0) # Fonctions vectorisees avec seulement f(x_gnom, y_gnom), les parametres sont deja remplaces.
        self._parameters_memory = {} # Permet d'eviter de relire le dictionaire des parametres a chaque fois.

    def cam_to_gnomonic(self, pxl_x, pxl_y, parameters, *, dtype=np.float32):
        """
        ** Passe des points de la camera dans un plan gnomonic. **

        Notes
        -----
        * Cette methode va 1.8 a 3.3 fois plus vite que celle de LaueTools.
        * Contrairement a LaueTools, cette methode prend en charge les vecteurs a n dimenssions.

        Parameters
        ----------
        pxl_x : float, int ou np.ndarray
            Coordonnee.s du.des pxl.s selon l'axe x dans le repere de la camera. (en pxl)
        pxl_y : float, int ou np.ndarray
            Coordonnee.s du.des pxl.s selon l'axe y dans le repere de la camera. (en pxl)
        parameters : dict
            Le dictionaire issue de la fonction ``laue.utilities.parsing.extract_parameters``.
        dtype : type, optional
            Si l'entree est un nombre et non pas une array numpy. Les calculs sont fait en ``float``.
            La representation machine des nombres. Par defaut ``np.float32`` permet des calculs rapide
            mais peu precis. Pour la presision il faut utiliser ``np.float64`` ou ``np.float128``.

        Returns
        -------
        float ou np.ndarray
            Le.s coordonnee.s x puis y du.des point.s dans le plan gnomonic eprimee.s en mm.
            Les dimenssions du tableau de sortie sont les memes que celle du tableau d'entree.
            shape de sortie = (2, *shape_d_entree)

        Examples
        -------
        >>> import numpy as np
        >>> from laue import Transformer
        >>> from laue.utilities.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, size=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>> x_cam, y_cam = np.linspace(3, 2048, 6), np.linspace(3, 2048, 6)
        >>>

        Output type
        >>> type(transformer.cam_to_gnomonic(x_cam, y_cam, parameters))
        <class 'numpy.ndarray'>
        >>> np.round(transformer.cam_to_gnomonic(x_cam, y_cam, parameters), 2)
        array([[-0.51, -0.36, -0.12,  0.1 ,  0.17,  0.13],
               [ 0.4 ,  0.32,  0.14, -0.18, -0.58, -0.94]], dtype=float32)
        >>> np.round(transformer.cam_to_gnomonic(x_cam, y_cam, parameters, dtype=np.float64), 2)
        array([[-0.51, -0.36, -0.12,  0.1 ,  0.17,  0.13],
               [ 0.4 ,  0.32,  0.14, -0.18, -0.58, -0.94]])
        >>>
        
        Output shape
        >>> transformer.cam_to_gnomonic(0.0, 0.0, parameters).shape
        (2,)
        >>> x_cam, y_cam = (np.random.uniform(0, 2048, size=(1, 2, 3)),
        ...                 np.random.uniform(0, 2048, size=(1, 2, 3)))
        >>> transformer.cam_to_gnomonic(x_cam, y_cam, parameters).shape
        (2, 1, 2, 3)
        >>>
        """
        assert isinstance(pxl_x, (float, int, np.ndarray)), \
            f"'pxl_x' can not be of type {type(pxl_x).__name__}."
        assert isinstance(pxl_y, (float, int, np.ndarray)), \
            f"'pxl_y' can not be of type {type(pxl_y).__name__}."
        assert type(pxl_x) == type(pxl_y), \
            f"Les 2 types sont differents: {type(pxl_x).__name__} vs {type(pxl_y).__name__}."
        if isinstance(pxl_x, np.ndarray):
            assert pxl_x.shape == pxl_y.shape, \
                f"Ils n'ont pas le meme taille: {pxl_x.shape} vs {pxl_y.shape}."
        assert isinstance(parameters, dict), ("Les parametres doivent founis "
            f"dans un dictionaire, pas dans un {type(parameters).__name__}")
        assert set(parameters) == {"dd", "xbet", "xgam", "xcen", "ycen", "pixelsize"}, \
            ("Les clefs doivent etres 'dd', 'xbet', 'xgam', 'xcen', 'ycen' et 'pixelsize'. "
            f"Or les clefs sont {set(parameters)}.")
        assert all(isinstance(v, numbers.Number) for v in parameters.values()), \
            "La valeurs des parametres doivent toutes etre des nombres."
        assert dtype in {np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64, np.float128. Pas {dtype}."

        if isinstance(pxl_x, np.ndarray):
            pxl_x, pxl_y = pxl_x.astype(dtype, copy=False), pxl_y.astype(dtype, copy=False)
        else:
            pxl_x, pxl_y = dtype(pxl_x), dtype(pxl_y)
        parameters = {k: dtype(v) for k, v in parameters.items()}

        hash_param = self._hash_parameters(parameters) # Recuperation de la 'signature' des parametres.
        optimized_func = self._fcts_cam_to_gnomonic[hash_param] # On regarde si il y a une fonction deja optimisee.

        if isinstance(optimized_func, int): # Si il n'y a pas de fonction optimisee.
            nbr_access = optimized_func # Ce qui est enregistre et le nombre de fois que l'on a chercher a y acceder.
            self._fcts_cam_to_gnomonic[hash_param] += 1 # Comme on cherche a y acceder actuelement, on peut incrementer le compteur.
            if nbr_access + 1 == 4: # Si c'est la 4 eme fois qu'on accede a la fonction.
                self.compile(parameters) # On optimise la fonction.
            else: # Si ce n'est pas encore le moment de perdre du temps a optimiser.
                return np.stack(self.get_fct_cam_to_gnomonic()(pxl_x, pxl_y,
                    parameters["dd"], parameters["xcen"], parameters["ycen"],
                    parameters["xbet"], parameters["xgam"], parameters["pixelsize"]))

        return np.stack(self._fcts_cam_to_gnomonic[hash_param](pxl_x, pxl_y))

    def dist_line(self, theta_vect, dist_vect, x_vect, y_vect, *, dtype=np.float64):
        """
        ** Calcul les distances projetees des points sur une droite. **

        Notes
        -----
        Pour des raisons de performances, travail avec des float32.

        Parameters
        ----------
        theta_vect : np.ndarray
            Les angles des droites normales aux droites principales.
            shape = ``(*nbr_droites)``
        dist_vect : np.ndarray
            Les distances entre les droites et l'origine.
            shape = ``(*nbr_droites)``
        x_vect : np.ndarray
            L'ensemble des coordonnees x des points.
            shape = ``(*nbr_points)``
        y_vect : np.ndarray
            L'ensemble des coordonnees y des points.
            shape = ``(*nbr_points)``
        dtype : type, optional
            La representation machine des nombres.
            Attention pour les calcul en float32 et moins
            risque d'y avoir des arrondis qui engendrent:
            ``RuntimeWarning: invalid value encountered in sqrt``.

        Returns
        -------
        np.ndarray
            Les distances des projetees des points sur chacunes des droites.
            shape = ``(*nbr_droites, *nbr_points)``

        Examples
        --------
        >>> import numpy as np
        >>> from laue import Transformer
        >>> from laue.utilities.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, size=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>>
        >>> lines = (np.array([0, np.pi/2]), np.array([1, 1])) # Horizontale et verticale passant par (1, 1)
        >>> points = (np.array([0, 1, 3, 0]), np.array([0, 1, 3, 1])) # Le points (0, 1), ...
        >>> np.round(transformer.dist_line(*lines, *points))
        array([[1., 0., 2., 1.],
               [1., 0., 2., 0.]])
        >>> np.round(transformer.dist_line(*lines, *points, dtype=np.float32))
        array([[1., 0., 2., 1.],
               [1., 0., 2., 0.]], dtype=float32)
        >>>
        >>> theta_vect, dist_vect = np.random.normal(size=(1, 2)), np.random.normal(size=(1, 2))
        >>> x_vect, y_vect = np.random.normal(size=(3, 4, 5)), np.random.normal(size=(3, 4, 5))
        >>> transformer.dist_line(theta_vect, dist_vect, x_vect, y_vect).shape
        (1, 2, 3, 4, 5)
        >>>
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 parametres de droite doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert isinstance(x_vect, np.ndarray), \
            f"'x_vect' has to be of type np.ndarray, not {type(x_vect).__name__}."
        assert isinstance(y_vect, np.ndarray), \
            f"'y_vect' has to be of type np.ndarray, not {type(y_vect).__name__}."
        assert x_vect.shape == y_vect.shape, \
            f"Les 2 coordonnees des points doivent avoir la meme shape: {x_vect.shape} vs {y_vect.shape}."
        assert dtype in {np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64, np.float128. Pas {dtype}."

        theta_vect, dist_vect = theta_vect.astype(dtype, copy=False), dist_vect.astype(dtype, copy=False)
        x_vect, y_vect = x_vect.astype(dtype, copy=False), y_vect.astype(dtype, copy=False)

        nbr_droites = theta_vect.shape
        nbr_points = x_vect.shape

        # Ca ne vaut pas le coup de paralleliser car c'est tres rapide.
        func = self.get_fct_dist_line()
        result = np.array([func(theta, dist, x_vect, y_vect)
                           for theta, dist
                           in zip(theta_vect.ravel(), dist_vect.ravel())
                          ], dtype=dtype).reshape((*nbr_droites, *nbr_points))
        return np.nan_to_num(result, copy=False, nan=0.0)

    def gnomonic_to_cam(self, gnom_x, gnom_y, parameters, *, dtype=np.float32):
        """
        ** Passe des points du plan gnomonic vers la camera. **

        Parameters
        ----------
        gnom_x : float ou np.ndarray
            Coordonnee.s du.des point.s selon l'axe x du repere du plan gnomonic. (en mm)
        gnom_y : float ou np.ndarray
            Coordonnee.s du.des point.s selon l'axe y du repere du plan gnomonic. (en mm)
        parameters : dict
            Le dictionaire issue de la fonction ``laue.utilities.parsing.extract_parameters``.
        dtype : type, optional
            Si l'entree est un nombre et non pas une array numpy. Les calculs sont fait en ``float``.
            La representation machine des nombres. Par defaut ``np.float32`` permet des calculs rapide
            mais peu precis. Pour la presision il faut utiliser ``np.float64`` ou ``np.float128``.

        Returns
        -------
        coords : np.ndarray
            * Le.s coordonnee.s x puis y du.des point.s dans le plan de la camera. (en pxl)
            * shape = (2, ...)

        Examples
        -------
        >>> import numpy as np
        >>> from laue import Transformer
        >>> from laue.utilities.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, size=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>> x_gnom, y_gnom = np.array([[-0.51176567, -0.35608186, -0.1245152 ,
        ...                              0.09978235,  0.17156848,  0.13417314 ],
        ...                            [ 0.40283853,  0.31846303,  0.14362221, 
        ...                             -0.18308422, -0.58226374, -0.93854752 ]])
        >>>

        Output type
        >>> type(transformer.gnomonic_to_cam(x_gnom, y_gnom, parameters))
        <class 'numpy.ndarray'>
        >>> np.round(transformer.gnomonic_to_cam(x_gnom, y_gnom, parameters))
        array([[   3.,  412.,  821., 1230., 1639., 2048.],
               [   3.,  412.,  821., 1230., 1639., 2048.]], dtype=float32)
        >>> np.round(transformer.gnomonic_to_cam(x_gnom, y_gnom, parameters, dtype=np.float64))
        array([[   3.,  412.,  821., 1230., 1639., 2048.],
               [   3.,  412.,  821., 1230., 1639., 2048.]])
        >>>
        
        Output shape
        >>> transformer.gnomonic_to_cam(0.0, 0.0, parameters).shape
        (2,)
        >>> x_cam, y_cam = (np.random.uniform(-.1, .1, size=(1, 2, 3)),
        ...                 np.random.uniform(-.1, .1, size=(1, 2, 3)))
        >>> transformer.gnomonic_to_cam(x_cam, y_cam, parameters).shape
        (2, 1, 2, 3)
        >>>
        """
        assert isinstance(gnom_x, (float, int, np.ndarray)), \
            f"'gnom_x' can not be of type {type(gnom_x).__name__}."
        assert isinstance(gnom_y, (float, int, np.ndarray)), \
            f"'gnom_y' can not be of type {type(gnom_y).__name__}."
        assert type(gnom_x) == type(gnom_y), \
            f"Les 2 types sont differents: {type(gnom_x).__name__} vs {type(gnom_y).__name__}."
        if isinstance(gnom_x, np.ndarray):
            assert gnom_x.shape == gnom_y.shape, \
                f"Ils n'ont pas la meme taille: {gnom_x.shape} vs {gnom_y.shape}."
        assert isinstance(parameters, dict), ("Les parametres doivent founis "
            f"dans un dictionaire, pas dans un {type(parameters).__name__}")
        assert set(parameters) == {"dd", "xbet", "xgam", "xcen", "ycen", "pixelsize"}, \
            ("Les clefs doivent etres 'dd', 'xbet', 'xgam', 'xcen', 'ycen' et 'pixelsize'. "
            f"Or les clefs sont {set(parameters)}.")
        assert all(isinstance(v, numbers.Number) for v in parameters.values()), \
            "La valeurs des parametres doivent toutes etre des nombres."
        assert dtype in {np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64, np.float128. Pas {dtype}."

        if isinstance(gnom_x, np.ndarray):
            gnom_x, gnom_y = gnom_x.astype(dtype, copy=False), gnom_y.astype(dtype, copy=False)
        else:
            gnom_x, gnom_y = dtype(gnom_x), dtype(gnom_y)
        parameters = {k: dtype(v) for k, v in parameters.items()}

        hash_param = self._hash_parameters(parameters) # Recuperation de la 'signature' des parametres.
        optimized_func = self._fcts_gnomonic_to_cam[hash_param] # On regarde si il y a une fonction deja optimisee.

        if isinstance(optimized_func, int): # Si il n'y a pas de fonction optimisee.
            nbr_access = optimized_func # Ce qui est enregistre et le nombre de fois que l'on a chercher a y acceder.
            self._fcts_gnomonic_to_cam[hash_param] += 1 # Comme on cherche a y acceder actuelement, on peut incrementer le compteur.
            if nbr_access + 1 == 4: # Si c'est la 4 eme fois qu'on accede a la fonction.
                self.compile(parameters) # On optimise la fonction.
            else: # Si ce n'est pas encore le moment de perdre du temps a optimiser.
                return np.stack(self.get_fct_gnomonic_to_cam()(gnom_x, gnom_y,
                    parameters["dd"], parameters["xcen"], parameters["ycen"],
                    parameters["xbet"], parameters["xgam"], parameters["pixelsize"]))

        return np.stack(self._fcts_gnomonic_to_cam[hash_param](gnom_x, gnom_y))

    def hough(self, x_vect, y_vect, *, dtype=np.float64):
        r"""
        ** Transformee de hough avec des droites. **

        Note
        ----
        * Pour des raisons de performances, les calculs se font sur des float32.
        * Les indices sont agences selon l'ordre defini par la fonction ``comb2ind``.

        Parameters
        ----------
        x_vect : np.ndarray
            L'ensemble des coordonnees x des points de shape: (*over_dims, nbr_points)
        y_vect : np.ndarray
            L'ensemble des coordonnees y des points de shape: (*over_dims, nbr_points)
        dtype : type, optional
            La representation machine des nombres.
            Attention pour les calculs en float32 et moins
            risque d'y avoir des arrondis qui engendrent:
            ``RuntimeWarning: invalid value encountered in sqrt``.

        Returns
        -------
        np.ndarray
            * theta : np.ndarray
                * Les angles au sens trigomometrique des vecteurs reliant l'origine
                ``O`` (0, 0) au point ``P`` appartenant a la droite tel que ``||OP||``
                soit la plus petite possible.
                * theta € [-pi, pi]
                * shape = ``(*over_dims, n*(n-1)/2)``
            * dist : np.ndarray
                * Ce sont les normes des vecteur ``OP``.
                * dist € [0, +oo].
                * shape = ``(*over_dims, n*(n-1)/2)``
            * Ces 2 grandeurs sont concatenees dans une seule array de
        shape = ``(2, *over_dims, n*(n-1)/2)``

        Examples
        --------
        >>> import numpy as np
        >>> from laue.core.geometry import transformer
        >>> transformer = transformer.Transformer()
        >>> x, y = np.random.normal(size=(2, 6))
        >>> transformer.hough(x, y).shape
        (2, 15)
        >>>
        >>> x, y = np.random.normal(size=(2, 4, 5, 6))
        >>> transformer.hough(x, y).shape
        (2, 4, 5, 15)
        >>> 
        """
        assert isinstance(x_vect, np.ndarray), \
            f"'x_vect' has to be of type np.ndarray, not {type(x_vect).__name__}."
        assert isinstance(y_vect, np.ndarray), \
            f"'y_vect' has to be of type np.ndarray, not {type(y_vect).__name__}."
        assert x_vect.shape == y_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {x_vect.shape} vs {y_vect.shape}."
        assert dtype in {np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64, np.float128. Pas {dtype}."

        n = x_vect.shape[-1]
        if n == 1:
            over_dims = x_vect.shape[:-1]
            clusters = np.empty(np.prod(over_dims, dtype=int), dtype=object)
            clusters[:] = [[] for _ in range(clusters.size)]
            clusters = clusters.reshape(over_dims)
            return clusters
        
        x_vect, y_vect = x_vect.astype(dtype, copy=False), y_vect.astype(dtype, copy=False)

        xa = np.concatenate([np.repeat(x_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        ya = np.concatenate([np.repeat(y_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        xb = np.concatenate([x_vect[..., i+1:] for i in range(n-1)], axis=-1)
        yb = np.concatenate([y_vect[..., i+1:] for i in range(n-1)], axis=-1)

        return np.nan_to_num(
            np.stack(self.get_fct_hough()(xa, ya, xb, yb)),
            copy=False,
            nan=0.0)

    def hough_reduce(self, theta_vect, dist_vect, *, nbr=4, tol=0.018, dtype=np.float32):
        """
        ** Regroupe des droites ressemblantes. **

        Notes
        -----
        * Cette methode est concue pour traiter les donnees issues de ``laue.core.geometry.transformer.Transformer.hough``.
        * La metrique utilise est la distance euclidiene sur un cylindre ferme sur theta.
        * En raison de performance et de memoire, les calculs se font sur des float32.

        Parameters
        ----------
        theta_vect : np.ndarray
            * Vecteur des angles compris entre [-pi, pi].
            * shape = ``(*over_dims, nbr_inter)``
        dist_vect : np.ndarray
            * Vecteur des distances des droites a l'origine comprises [0, +oo].
            * shape = ``(*over_dims, nbr_inter)``
        tol : float
            La distance maximal separant 2 points dans l'espace de hough reduit,
            (ie la difference entre 2 droites dans l'espace spacial) tel que les points
            se retrouvent dans le meme cluster. Plus ce nombre est petit, plus les points
            doivent etre bien alignes. C'est une sorte de tolerance sur l'alignement.
        nbr : int
            C'est le nombre minimum de points presque alignes pour que
            l'on puisse considerer la droite qui passe par ces points.
            Par defaut, les droites qui ne passent que par 4 points et plus sont retenues.
        dtype : type, optional
            La representation machine des nombres. Par defaut ``np.float32`` permet des calculs rapide
            mais peu precis. Pour la presision il faut utiliser ``np.float64``.
            ``np.float128`` est interdit car c'est un peu over-kill pour cette methode!

        Returns
        -------
        np.ndarray(dtype=float), np.ndarray(dtype=object)
            * Ce sont les centres des clusters pour chaque 'nuages de points'. Cela correspond
            aux angles et aux distances qui caracterisent chaque droites.
            * Si les parametres d'entres sont des vecteurs 1d, le resultat sera une array
            numpy contenant les **angles** puis les **distances**. Donc de shape = ``(2, nbr_clusters)``
            * Si les parametres d'entres sont en plusieur dimensions, (representes plusieur
            nuages de points indepandant), alors le resultat sera une array d'objet de
            shape = ``(*over_dims)``. Chaque objet est lui meme un array, resultat recursif
            de l'appel de cette fonction sur le nuage de points unique correspondant.

        Examples
        --------
        >>> import numpy as np
        >>> from laue.core.geometry import transformer
        >>> transformer = transformer.Transformer()

        Type de retour ``float`` vs ``object``.
        >>> x, y = (np.array([ 1.,  2.,  3.,  0., -1.]),
        ...         np.array([ 0.,  1.,  1., -1.,  1.]))
        >>> theta, dist = transformer.hough(x, y)
        >>> np.round(transformer.hough_reduce(theta, dist, nbr=3), 2)
        array([[-0.79,  1.57],
               [ 0.71,  1.  ]], dtype=float32)
        >>> res = transformer.hough_reduce(theta.reshape((1, -1)), dist.reshape((1, -1)), nbr=3)
        >>> res.dtype
        dtype('O')
        >>> res.shape
        (1,)
        >>> np.round(res[0], 2)
        array([[-0.79,  1.57],
               [ 0.71,  1.  ]], dtype=float32)
        >>>

        Les dimensions de retour.
        >>> x, y = (np.random.normal(size=(6, 5, 4)),
        ...         np.random.normal(size=(6, 5, 4)))
        >>> theta, dist = transformer.hough(x, y)
        >>> transformer.hough_reduce(theta, dist).shape
        (6, 5)
        >>> 
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert theta_vect.ndim >= 1, "La matrice ne doit pas etre vide."
        assert isinstance(tol, float), f"'tol' has to be a float, not a {type(tol).__name__}."
        assert 0.0 < tol <= 0.5, ("Les valeurs coherentes de 'tol' se trouvent entre "
            f"]0, 1/2], or tol vaut {tol}, ce qui sort de cet intervalle.")
        assert isinstance(nbr, int), f"'nbr' has to be an integer, not a {type(nbr).__name__}."
        assert 2 < nbr, f"2 points sont toujours alignes! Vous ne pouvez pas choisir nbr={nbr}."
        assert dtype in {np.float16, np.float32, np.float64}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64. Pas {dtype}."

        # On fait la conversion des le debut pour un gain de temps.
        theta_vect, dist_vect = theta_vect.astype(dtype, copy=False), dist_vect.astype(dtype, copy=False)

        *over_dims, nbr_inter = theta_vect.shape # Recuperation des dimensions.
        nbr = (nbr*(nbr-1))/2 # On converti le nombre de points alignes en nbr de segments.

        # On commence par travailler avec les donnees reduites.
        theta_theo_std = math.pi / math.sqrt(3) # Variance theorique = (math.pi - -math.pi)**2 / 12
        dist_std = np.nanstd(dist_vect, axis=-1) # Ecart type non biaise (sum(*over_dims)/N), shape: (*over_dims)
        dist_vect = (dist_vect * theta_theo_std
            / np.repeat(dist_std[..., np.newaxis], nbr_inter, axis=-1)) # Les distances quasi reduites.
        
        # Extraction des clusters.
        if not len(over_dims): # Cas des tableaux 1d.
            return self._clustering_1d(theta_vect, dist_vect, dist_std, tol, nbr)

        clusters = np.empty(np.prod(over_dims, dtype=int), dtype=object) # On doit d'abord creer un tableau d'objet 1d.
        if multiprocessing.current_process().name == "MainProcess" and np.prod(over_dims) >= os.cpu_count(): # Si ca vaut le coup de parraleliser:
            ser_self = cloudpickle.dumps(self) # Strategie car 'pickle' ne sais pas faire ca.
            from laue.utilities.multi_core import pickleable_method
            with multiprocessing.Pool() as pool:
                clusters[:] = pool.map(
                    pickleable_method, # Car si il y a autant de cluster dans chaque image,
                    (                   # numpy aurait envi de faire un tableau 2d plutot qu'un vecteur de listes.
                        (
                            Transformer._clustering_1d,
                            ser_self,
                            {"theta_vect_1d":theta, "dist_vect_1d":dist, "std":std, "tol":tol, "nbr":nbr}
                        )
                        for theta, dist, std
                        in zip(
                            theta_vect.reshape((-1, nbr_inter)),
                            dist_vect.reshape((-1, nbr_inter)),
                            np.nditer(dist_std)
                        )
                    )
                )
        else:
            clusters[:] = [self._clustering_1d(theta, dist, std, tol, nbr)
                           for theta, dist, std in zip(
                                    theta_vect.reshape((-1, nbr_inter)),
                                    dist_vect.reshape((-1, nbr_inter)),
                                    np.nditer(dist_std))] 
        clusters = clusters.reshape(over_dims) # On redimensione a la fin de sorte a garentir les dimensions.

        return clusters

    def inter_lines(self, theta_vect, dist_vect, *, dtype=np.float32):
        r"""
        ** Calcul les points d'intersection entre les droites. **

        Notes
        -----
        * Cette methode est concue pour traiter les donnees issues de ``laue.core.geometry.transformer.Transformer.hough``.
        * En raison de performance et de memoire, les calculs se font sur des float32.
        * Les indices sont agences selon l'ordre defini par la fonction ``comb2ind``.

        Parameters
        ----------
        theta_vect : np.ndarray
            * Vecteur des angles compris entre [-pi, pi].
            * shape = (*over_dims, nbr_droites)
        dist_vect : np.ndarray
            * Vecteur des distances des droites a l'origine comprises [0, +oo].
            * shape = (*over_dims, nbr_droites)
        dtype : type, optional
            La representation machine des nombres. Par defaut
            ``np.float32`` permet des calculs rapide
            mais peu precis. Pour la presision il faut utiliser
            ``np.float64`` ou ``(getattr(np, "float128") if hasattr(np, "float128") else np.float64)``.

        Returns
        -------
        np.ndarray
            * Dans le dommaine spatial et non pas le domaine de hough, cherche
            les intersections des droites. Il y a ``n*(n-1)/2`` intersections, n etant
            le nombre de droites. donc la complexite de cette methode est en ``o(n**2)``.
            * Si les vecteurs d'entre sont des vecteurs 1d (ie ``*over_dims == ()``), 
            Seront retournes le vecteur d'intersection selon l'axe x et le vecteur
            des intersections selon l'axe y. Ces 2 vecteurs de meme taille sont concatenes
            sous la forme d'une matrice de shape = ``(2, n*(n-1)/2)``.
            * Si les vecteurs d'entre sont en plusieurs dimensions, seul les droites de la
            derniere dimensions se retrouvent dans la meme famille. Tous comme pour les
            vecteurs 1d, on trouve d'abord les intersections selon x puis en suite selon y.
            La shape du tenseur final est donc: ** shape = ``(2, *over_dims, n*(n-1)/2)`` **.

        Examples
        -------
        >>> import numpy as np
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> np.random.seed(0)
        >>> x, y = np.random.normal(size=(2, 4, 5, 6))
        >>> theta, dist = transformer.hough(x, y)
        >>> theta.shape
        (4, 5, 15)
        >>> transformer.inter_lines(theta, dist).shape
        (2, 4, 5, 105)
        >>>
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert theta_vect.ndim >= 1, "La matrice ne doit pas etre vide."
        assert theta_vect.shape[-1] >= 2, \
            f"Il doit y avoir au moins 2 droites par famille, pas {theta_vect.shape[-1]}."
        assert dtype in {np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)}, \
            f"Les types ne peuvent etre que np.float16, np.float32, np.float64, np.float128. Pas {dtype}."

        theta_vect, dist_vect = theta_vect.astype(dtype, copy=False), dist_vect.astype(dtype, copy=False)
        n = theta_vect.shape[-1]

        theta_1 = np.concatenate([np.repeat(theta_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        dist_1 = np.concatenate([np.repeat(dist_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        theta_2 = np.concatenate([theta_vect[..., i+1:] for i in range(n-1)], axis=-1)
        dist_2 = np.concatenate([dist_vect[..., i+1:] for i in range(n-1)], axis=-1)

        return np.stack(self.get_fct_inter_line()(theta_1, dist_1, theta_2, dist_2))

    def _clustering_1d(self, theta_vect_1d, dist_vect_1d, std, tol, nbr):
        """
        ** Help for hough_reduce. **

        * Permet de trouver les clusters d'un nuage de points.
        * La projection 3d, bien que moins realiste, est 20% plus rapide que la distance reele.
        """
        from sklearn.cluster import DBSCAN

        dtype_catser = theta_vect_1d.dtype.type
        THETA_STD = dtype_catser(math.pi / math.sqrt(3))
        WEIGHT = 0.65 # 0 => tres souple sur les angles, 1=> tres souple sur les distances.

        # On retire les droites aberantes.
        mask_to_keep = np.isfinite(theta_vect_1d) & np.isfinite(dist_vect_1d)
        if not mask_to_keep.any(): # Si il ne reste plus rien.
            return np.array([], dtype=dtype_catser)
        theta_vect_1d, dist_vect_1d = theta_vect_1d[mask_to_keep], dist_vect_1d[mask_to_keep]

        # On passe dans un autre repere de facon a ce que -pi et pi se retrouvent a cote.
        if numexpr is not None:
            theta_x = numexpr.evaluate("2*WEIGHT*cos(theta_vect_1d)")
            theta_y = numexpr.evaluate("2*WEIGHT*sin(theta_vect_1d)")
        else:
            theta_x, theta_y = 2*WEIGHT*np.cos(theta_vect_1d), 2*WEIGHT*np.sin(theta_vect_1d)

        # Recherche des clusters.
        n_jobs = -1 if multiprocessing.current_process().name == "MainProcess" else 1
        db_res = DBSCAN(eps=tol, min_samples=nbr, n_jobs=n_jobs).fit(
            np.vstack((theta_x, theta_y, 2*(1-WEIGHT)*dist_vect_1d)).transpose())

        # Mise en forme des clusters.
        clusters_dict = collections.defaultdict(lambda: [])
        keep = db_res.labels_ != -1 # Les indices des clusters a garder.
        for x_cyl, y_cyl, dist, group in zip(
                theta_x[keep], theta_y[keep], dist_vect_1d[keep], db_res.labels_[keep]):
            clusters_dict[group].append((x_cyl, y_cyl, dist))

        theta = np.array([np.arccos(cluster[:, 0].mean()/(2*WEIGHT))*np.sign(cluster[:, 1].sum())
                    for cluster in map(np.array, clusters_dict.values())],
                    dtype=dtype_catser)
        dist = np.array([cluster[:, 2].mean()
                        for cluster in map(np.array, clusters_dict.values())],
                    dtype=dtype_catser) * std / THETA_STD
        return np.array([theta, dist], dtype=dtype_catser)

    def _hash_parameters(self, parameters):
        """
        ** Hache le dictionaire des parametres. **

        * Il n'y a pas de verification pour des histoires de performances.

        Parameters
        ----------
        parameters : dict
            Dictionaire des parametres issues de ``laue.utilities.parsing.extract_parameters``.

        Returns
        -------
        int
            Un identifiant tq 2 dictionaires identiques renvoient le meme id.
        """
        return hash(( # Il faut imperativement garantir l'ordre.
            parameters["dd"],
            parameters["xcen"],
            parameters["ycen"],
            parameters["xbet"],
            parameters["xgam"],
            parameters["pixelsize"]))


def comb2ind(ind1, ind2, n):
    """
    ** Transforme 2 indices en un seul. **

    Note
    ----
    * Bijection de ``ind2comb``.
    * Peut etre utile pour les methodes ``laue.core.geometry.transformer.Transformer.hough``
    et ``laue.core.geometry.transformer.Transformer.inter_lines``.

    Parameters
    ----------
    ind1 : int ou np.ndarray(dtype=int)
        L'indice du premier element: ``0 <= ind1``.
    ind2 : int ou np.ndarray(dtype=int)
        L'indice du second element: ``ind1 < ind2``.
    n : int
        Le nombre de symboles : ``2 <= n and ind2 < n``.

    Returns
    -------
    int, np.ndarray(dtype=int)
        Le nombre de mots contenant exactement 2 symboles dans un
        alphabet de cardinal ``n``. Sachant que le deuxieme symbole
        est stricement superieur au premier, et que les mots sont
        generes avec le comptage naturel (representation des nombres
        en base n).

    Examples
    -------
    >>> from laue.core.geometry.transformer import comb2ind
    >>> comb2ind(0, 1, n=6)
    0
    >>> comb2ind(0, 2, n=6)
    1
    >>> comb2ind(0, 5, n=6)
    4
    >>> comb2ind(1, 2, n=6)
    5
    >>> comb2ind(4, 5, n=6)
    14
    """
    assert isinstance(ind1, (int, np.ndarray)), \
        f"'ind1' can not being of type {type(ind1).__name__}."
    assert isinstance(ind2, (int, np.ndarray)), \
        f"'ind2' can not being of type {type(ind2).__name__}."
    assert isinstance(n, int), f"'n' has to ba an integer, not a {type(n).__name__}."
    if isinstance(ind1, np.ndarray):
        assert ind1.dtype == int or issubclass(ind2.dtype.type, np.integer), \
            f"'ind1' must be integer, not {str(ind1.dtype)}."
        assert (ind1 >= 0).all(), "Tous les indices doivent etres positifs."
    else:
        assert ind1 >= 0, "Les indices doivent etre positifs."
    if isinstance(ind2, np.ndarray):
        assert ind2.dtype == int or issubclass(ind2.dtype.type, np.integer), \
            f"'ind2' must be integer, not {str(ind2.dtype)}."
        assert ind1.shape == ind2.shape, ("Si les indices sont des arrays, elles doivent "
            f"toutes 2 avoir les memes dimensions. {ind1.shape} vs {ind2.shape}.")
        assert (ind2 > ind1).all(), ("Les 2ieme indices doivent "
            "etres strictement superieur aux premiers.")
        assert (ind2 < n).all(), "Vous aimez un peu trop les 'index out of range'."
    else:
        assert ind2 > ind1, ("Le 2ieme indice doit etre strictement superieur au premier. "
            f"{ind1} vs {ind2}.")
        assert ind2 < n, "Vous aimez un peu trop les 'index out of range'."

    return n*ind1 - (ind1**2 + 3*ind1)//2 + ind2 - 1

def ind2comb(comb, n):
    """
    ** Eclate un rang en 2 indices **

    Notes
    -----
    * Bijection de ``comb2ind``.
    * Peut etre utile pour les methodes ``laue.core.geometry.transformer.Transformer.hough``
    et ``laue.core.geometry.transformer.Transformer.inter_lines``.
    * Risque de donner de faux resultats pour n trop grand.

    Parameters
    ----------
    comb : int ou np.ndarray(dtype=int)
        L'indice, ie le (comb)ieme mot constitue 2 de symbols
        ajence comme decrit dans ``comb2ind``.
    n : int
        Le nombre de symboles.

    Returns
    -------
    int ou np.ndarray(dtype=int)
        Le premier des 2 indicices (``ind1``).
    int ou np.ndarray(dtype=int)
        Le second des 2 indicices (``ind2``). De sorte que ``comb2ind(ind1, ind2) == comb``.

    Examples
    --------
    >>> from laue.core.geometry.transformer import ind2comb
    >>> ind2comb(0, n=6)
    (0, 1)
    >>> ind2comb(1, n=6)
    (0, 2)
    >>> ind2comb(4, n=6)
    (0, 5)
    >>> ind2comb(5, n=6)
    (1, 2)
    >>> ind2comb(14, n=6)
    (4, 5)
    >>>
    """
    assert isinstance(comb, (int, np.ndarray)), \
        f"'comb' can not being of type {type(comb).__name__}."
    assert isinstance(n, int), f"'n' has to ba an integer, not a {type(n).__name__}."
    if isinstance(comb, np.ndarray):
        assert comb.dtype == int or issubclass(comb.dtype.type, np.integer), f"'comb' must be integer, not {str(comb.dtype)}."
        assert (comb >= 0).all(), "Tous les indices doivent etres positifs."
        assert (comb < n*(n-1)/2).all(), (f"Dans un alphabet a {n} symboles, il ne peut y a voir "
            f"que {n*(n-1)/2} mots. Le mot d'indice {comb.max()} n'existe donc pas!")
    else:
        assert comb >= 0, "Les indices doivent etre positifs."
        assert comb < n*(n-1)/2, (f"Dans un alphabet a {n} symboles, il ne peut y a voir "
            f"que {n*(n-1)/2} mots. Le mot d'indice {comb} n'existe donc pas!")

    homogeneous_int = lambda x: x.astype(int) if isinstance(x, np.ndarray) else int(x)
    ind1 = homogeneous_int(np.ceil(n - np.sqrt(-8*comb + 4*n**2 - 4*n - 7)/2 - 3/2))
    ind2 = comb + (ind1**2 + 3*ind1)//2 - ind1*n + 1
    return ind1, ind2
