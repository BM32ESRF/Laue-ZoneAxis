#!/usr/bin/env python3

"""
** Permet de manipuler un diagramme de Laue unique. **
-----------------------------------------------------

Notes
-----
Si le module ``psutil`` est installe, la memoire sera mieux geree.
"""

import math
import os

import cv2
import numpy as np
try:
    import psutil # Pour acceder a la memoire disponible.
except ImportError:
    psutil = None

from laue.spot import Spot
from laue.tools.splitable import Splitable


__pdoc__ = {"LaueDiagram.__contains__": True,
            "LaueDiagram.__getitem__": True,
            "LaueDiagram.__hash__": True,
            "LaueDiagram.__iter__": True,
            "LaueDiagram.__len__": True}


class LaueDiagram(Splitable):
    """
    Represente un diagramme de Laue associe a une seule image.
    """
    def __init__(self, name, spots, experiment, **kwargs):
        """
        Notes
        -----
        * L'utilisateur n'a pas a generer des objets issus de cette classe.
        Ils sont generes automatiquement par des instances de ``laue.experiment.base_experiment.Experiment``.
        * Il n'y a pas de verifications faites sur les entrees car l'utilisateur
        ne doit pas toucher a l'initialisateur. La performance passe donc avant
        l'enorme mefiance envers les humains.

        Parameters
        ----------
        name : str
            Nom de l'image du diagramme.
        spots : set
            Ensemble des points chauds. Ils doivent heriter de ``laue.spot.Spot``.
        experiment : Experiment
            Instance de l'experience qui contient ce diagramme.
            Cet objet doit heriter de ``laue.experiment.base_experiment.Experiment``.
        image_xy : np.ndarray, optional
            Matrice numpy de l'image de depart.
        """
        self.name = name
        self.experiment = experiment # C'est l'experience qui contient ce diagramme.
        self.spots = spots # La liste des spots en vrac. Pas set car l'ordre doit etre fige.
        self.image_xy = kwargs.get("image_xy", None) # None si il faut preserver la RAM.

        # Declaration des variables futur.
        self.quality = None # Facteur qui dit a quel point ce diagramme est joli a l'oeil.
        self.image_gnom = None # Image projete dans le plan gnomonic.
        self.sorted_spots = {} # Les listes des spots tries selon un ordre particulier.
        self.axis = {} # Les axes de zones
        self.spots_set = None # L'ensemble des spots pour une recherche plus rapide.

    def find_zone_axes(self, *, dmax=None, nbr=7, tol=None,
        _axes_args=None, _get_args=False):
        """
        ** Cherche les axes de zone **

        Notes
        -----
        Si le but est d'extraire les axes de zonnes de plusieurs diagrammes
        il vaut mieux appeler ``laue.experiment.base_experiment.Experiment.find_zone_axes`` car
        les calculs sont parallelises, contrairement a cette methode.

        Parameters
        ----------
        dmax : float, optional
            La distance maximale admissible entre un spot et un axe de zone
            pour pouvoir considerer que le spot appartient a l'axe de zone.
            Par defaut cette valeur evolue lineairement entre 5 pxl pour
            les diagrammes contenants beaucoup de spots a 20 pxl pour les petits.
            avec ``n`` le nombre de spots dans le diagramme.
        tol : float, optional
            Alignement des points. Voir ``laue.geometry.Transformer.hough_reduce``
            pour avoir les informations precises sur 'tol'. Par defaut
            cette valeur evolue exponentiellement entre 0.018 pour les diagrammes
            de 50 spots et 0.005 pour ceux de 600 spots.
        nbr : int, optional
            Nombre minimum de points par axe de zone.

        Returns
        -------
        axis : list
            La liste des axes de zone de type ``laue.zone_axis.ZoneAxis``.

        Examples
        -------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> type(diag.find_zone_axes())
        <class 'list'>
        >>> type(diag.find_zone_axes().pop())
        <class 'laue.zone_axis.ZoneAxis'>
        >>>
        """
        if dmax is None:
            pxl_max, pxl_min = 20, 5
            d_max, d_min = 1.2*pxl_max/2048, 1.2*pxl_min/2048
            dmax = max(.005, d_max - (d_max-d_min)/800 * len(self))
        if tol is None:
            a = 2.2865e-8
            b = -3.9201e-5
            c = .0205058
            tol = a*len(self)**2 + b*len(self) + c

        assert isinstance(dmax, float), \
            f"'dmax' doit etre un flottant, pas un {type(dmax).__name__}."
        assert dmax > 0, f"La distance doit etre strictement positive elle vaut {dmax}."

        if _get_args: # Si il faut seulement preparer le travail.
            gnomonics = self.get_gnomonic_positions()
            return gnomonics, dmax, nbr, tol

        if (dmax, nbr, tol) in self.axis: # Si on a deja la solution.
            return self.axis[(dmax, nbr, tol)]

        if _axes_args is None: # Si le travail n'est pas premache.
            from laue.zone_axis import _get_zone_axes_pickle
            angles, dists, axis_spots_ind, spots_axes_ind = _get_zone_axes_pickle(
                (self.experiment.transformer,
                self.get_gnomonic_positions(),
                dmax, nbr, tol))
        else:
            angles, dists, axis_spots_ind, spots_axes_ind = _axes_args

        # Creation des objets 'ZoneAxis'.
        from laue.zone_axis import ZoneAxis
        self.axis[(dmax, nbr, tol)] = [
                ZoneAxis(diagram=self,
                     spots_ind=spots_ind,
                     identifier=i,
                     angle=angle,
                     dist=dist)
                for i, (angle, dist, spots_ind) in enumerate(zip(angles, dists, axis_spots_ind))]

        # Attribution des axes aux spots.
        for spot, axes_ind in zip(self, spots_axes_ind):
            spot.axes = {self.axis[(dmax, nbr, tol)][axis_ind] for axis_ind in axes_ind}
        return self.axis[(dmax, nbr, tol)]

    def get_image_gnomonic(self):
        """
        ** Recupere le contenu de l'image d'un diagramme projete dans le plan gnomonic. **

        Notes
        -----
        Les parametres de set_calibration de la camera sont recuperes avec
        un appel a la fonction ``laue.experiment.base_experiment.Experiment.set_calibration()``.

        Returns
        -------
        image: np.ndarray(dtype=np.uint16)
            L'image 2d en niveau de gris encodee en uint16.
        
        Raises
        ------
        NameError
            Si l'image est introuvable.
        AttributError
            Si il manque des infos pour satisfaire cette demande.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> diag.get_image_gnomonic()
        array([[0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0]], dtype=uint16)
        >>>
        """
        # Interpolation inverse vers l'image finale.
        map_x, map_y, _ = self.experiment._get_gnomonic_matrix()
        image = cv2.remap(self.get_image_xy(),
            map_x, map_y,
            interpolation=cv2.INTER_CUBIC)

        if psutil is not None and psutil.virtual_memory().percent < 50:
            self.image_gnom = image

        return image

    def get_image_xy(self):
        """
        ** Recupere le contenu de l'image d'un diagramme. **

        Returns
        -------
        image : np.ndarray(dtype=np.uint16)
            L'image 2d en niveau de gris encodee en uint16.
        
        Raises
        ------
        NameError
            Si l'image est introuvable.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> diag.get_image_xy()
        array([[0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               ...,
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0],
               [0, 0, 0, ..., 0, 0, 0]], dtype=uint16)
        >>>
        """
        if self.image_xy is not None:
            return self.image_xy

        if not os.path.exists(self.get_id()):
            raise NameError(f"Impossible de trouver le fichier {repr(self.get_id())}.")

        from laue.tools.image import read_image
        image = read_image(self.get_id())

        if psutil is not None and psutil.virtual_memory().percent < 50:
            self.image_xy = image

        return image

    def get_id(self):
        """
        ** Retourne le nom du diagramme. **

        * Dans la mesure du possible, le nom du diagramme est le chemin
        d'acces au fichier image qui a permis de constituer le diagramme.
        * Si le chemin d'acces est inconnu, un nom par defaut unique est genere.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> diag.get_id()
        'laue/examples/ge_blanc.mccd'
        >>>
        """
        return self.name

    def select_spots(self, *, n=None, sort=None):
        """
        ** Recupere une partie des spots. **

        Notes
        -----
        Les pointeurs des spots renvoyes sont dupliques, c'est une copie superficielle.
        Une suppression ou un ajout de spot dans la liste ne changera pas le diagramme
        par contre une modification d'un attribut d'un des spots va etre effectif,
        et modifira donc definitivement le spot considere.

        Parameters
        ----------
        n : int, optional
            Nombre de spots a considerer. La valeur ``None`` indique
            que tous les spots sont renvoyes.
        sort : str or callable, optional
            - None => Les spots ne sont pas tries (le plus rapide). Ils sont cedes dans
            un ordre quelquonque mais systematique. L'ordre reste inchange entre 2 appels.
            - callable => Clef de tri, qui a chaque spot de type ``laue.spot.Spot``.
            associe un flotant. Les spots ayant des petits flottant se retrouveront
            au debut, ceux avec un gros seront en fin de chaine.
            - str => La methode de tri. Il y en a plusieurs possibles:
                - "intensity" => Les spots sont renvoyes par intensite decroissante.
                - "distortion" => Les spots sont renvoyes par distortion croissante.
                - "quality" => Les spots sont renvoyes par qualite decroissante.

        Returns
        -------
        list
            La liste des spots. Chaque element est de type ``laue.spot.Spot``.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> diag.select_spots(n=2, sort="intensity")
        [Spot(bbox=(617, 1651, 11, 13), distortion=1.0760), Spot(bbox=(928, 1210, 10, 11), distortion=1.0673)]
        >>> diag.select_spots(n=2, sort="distortion")
        [Spot(bbox=(928, 1210, 10, 11), distortion=1.0673), Spot(bbox=(617, 1651, 11, 13), distortion=1.0760)]
        >>> diag.select_spots(n=2, sort=lambda spot: spot.get_position()[0])
        [Spot(bbox=(130, 1202, 5, 6), distortion=1.1804), Spot(bbox=(157, 905, 7, 6), distortion=1.1342)]
        >>>
        """
        assert n is None or isinstance(n, int), f"'n' can not be {type(n).__name__}."
        assert n is None or n > 0, f"'n' can not be {n}."
        assert (sort is None or hasattr(sort, "__call__")
            or sort in {"intensity", "distortion", "quality"}), \
            f"'sort' ne peut pas etre {sort}."

        if sort is None: # Si il n'y a pas de tri a faire.
            if n is None:
                return self.spots.copy()
            return self.spots[:n]

        if hasattr(sort, "__call__"):
            l_spots = sorted(self.spots, key=sort)
        if sort in self.sorted_spots: # On enregistre la liste pour de melleur
            l_spots = self.sorted_spots[sort] #  perfs aux apels suivants.
        else:
            if sort == "intensity":
                l_spots = sorted(self.spots, key=(lambda spot: -spot.get_intensity()))
            elif sort == "distortion":
                l_spots = sorted(self.spots, key=(lambda spot: spot.get_distortion()))
            elif sort == "quality":
                l_spots = sorted(self.spots, key=(lambda spot: -spot.get_quality()))
            self.sorted_spots[sort] = l_spots

        if n is not None:
            return l_spots[:n]
        return l_spots

    def get_positions(self, *, n=None, sort=None):
        """
        ** Recupere la position des spots dans le plan de la camera. **

        Parameters
        ----------
        n : int, optional
            Same as ``LaueDiagram.select_spots``.
        sort : str or callable, optional
            Same as ``LaueDiagram.select_spots``.

        Returns
        -------
        np.ndarray
            * Le vecteur des coordonnees x puis le vecteur des y. (en pxl)
            * La shape de retour est (2, nbr_spots).
            * Les spots ne sont pas tries, l'ordre est le meme que
            ``LaueDiagram.select_spots`` sans argument.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> diag.get_positions().shape
        (2, 78)
        >>> diag.get_positions(n=4, sort=lambda spot: spot.get_position()[0])
        array([[ 132.0286 ,  160.35379,  192.01744,  214.02731],
               [1204.656  ,  907.2095 , 1296.9255 ,  492.425  ]], dtype=float32)
        >>>
        """
        return np.array(
            [spot.get_position() for spot in self.select_spots(n=n, sort=sort)],
            dtype=np.float32).transpose()

    def get_gnomonic_positions(self, *, n=None, sort=None):
        """
        ** Recupere la position des spots dans le plan gnomonic. **

        Parameters
        ----------
        n : int, optional
            Same as ``LaueDiagram.select_spots``.
        sort : str or callable, optional
            Same as ``LaueDiagram.select_spots``.

        Returns
        -------
        coordonees : np.ndarray
            * Le vecteur des coordonnees x puis le vecteur y. (en mm)
            * La shape de retour est (2, nbr_spots)

        Raises
        ------
        AttributError
            Si il manque des infos pour satisfaire cette demande.
            En general l'un des parametres de set_calibration.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> diag.get_gnomonic_positions().shape
        (2, 78)
        >>> diag.get_gnomonic_positions(n=4, sort=lambda spot: spot.get_position()[0])
        array([[-0.0978478 , -0.20958039, -0.04084784, -0.3524825 ],
               [ 0.5766271 ,  0.48377115,  0.56894475,  0.36476415]],
              dtype=float32)
        >>>
        """
        # On calcul les projections pour tous les points a la fois.
        if self.spots[0].gnomonic is None:
            coord_gnomonic = self.experiment.transformer.cam_to_gnomonic(
                *self.get_positions(n=n, sort=sort),
                self.experiment.set_calibration())
            for spot, xg, yg in zip(self, *coord_gnomonic):
                spot.gnomonic = (xg, yg)
        # On extrait juste ce qu'il nous interresse.
        else:
            coord_gnomonic = np.array(
                [spot.get_gnomonic() for spot in self.select_spots(n=n, sort=sort)],
                dtype=np.float32).transpose()

        return coord_gnomonic

    def get_quality(self):
        r"""
        ** Estime a quel point le diagramme est joli. **

        Returns
        -------
        quality : float
            * Un scalaire qui permet de juger de la purete du diagramme:
            * < 1 => diagramme tres moche, illisible a l'oeil.
            * < 2 => diagramme pas bien joli.
            * < 3 => diagramme bien joli, avec de belles taches.
            * \> 3 => diagramme super joli, bien epure avec des taches rondes et intenses.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> print(f"quality: {diag.get_quality():.4f}")
        quality: 1.2752
        >>>
        """
        def f_nbr(x, n_best_min, n_best_max):
            if x < n_best_min:
                return x / n_best_min
            if n_best_min <= x < n_best_max:
                return 1
            return math.exp(-(x-n_best_max)*(math.log(2)/n_best_max))

        if self.quality is not None:
            return self.quality

        spot_quality = np.mean([spot.get_quality() for spot in self])
        self.quality = f_nbr(len(self), 60, 120) * spot_quality
        return self.quality

    def find_subsets(self, *args, **kwargs):
        """
        ** Alias to ``laue.tools.splitable.Splitable.find_subsets``. **

        C'est une methode abstraite definie dans la classe mere.
        """
        return super().find_subsets(*args, **kwargs)

    def plot_all(self, *, display=True):
        """
        ** Affiche le diagramme a l'ecran. **

        * Utilise le module ``matplotlib`` qui doit etre installe.
        * Cette methode peut prendre du temps car elle affiche le maximum de choses possible.

        Parameters
        ----------
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Returns
        -------
        matplotlib.figure.Figure
            La figure matplotlib completee.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>> diag.plot_all(display=False)
        <Figure size 640x480 with 2 Axes>
        >>>
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.suptitle(self.get_id())
        axe_xy = fig.add_subplot(1, 2, 1)
        axe_gnomonic = fig.add_subplot(1, 2, 2)

        self.plot_xy(axe_xy, display=False)

        try:
            self.plot_gnomonic(axe_gnomonic, display=False)
        except AttributeError:
            pass

        if display:
            plt.show()

        return fig

    def plot_gnomonic(self, axe_pyplot=None, *, display=True):
        """
        ** Prepare l'affichage du diagramme dans le plan gnomonic. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte les methodes ``.scatter`` et ``.imshow``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>>
        >>> diag.plot_gnomonic(display=False)
        <AxesSubplot:title={'center':'plan gnomonic'}, xlabel='x.Gi (mm)', ylabel='y.Gj (mm)'>
        >>>
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> diag.plot_gnomonic(axe, display=False)
        <AxesSubplot:title={'center':'plan gnomonic'}, xlabel='x.Gi (mm)', ylabel='y.Gj (mm)'>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()

        axe_pyplot.set_title("plan gnomonic")
        axe_pyplot.set_xlabel("x.Gi (mm)")
        axe_pyplot.set_ylabel("y.Gj (mm)")

        # Affichage image de fond.
        try:
            image = self.get_image_gnomonic()
        except (NameError, AttributeError):
            pass
        else:
            *_, limits = self.experiment._get_gnomonic_matrix()
            mean, std = image.mean(), image.std()
            x_coords, y_coords = self.get_gnomonic_positions()
            axe_pyplot.imshow(image,
                origin='lower',
                aspect=((self.experiment.get_images_shape()[1]*x_coords.ptp())
                      / (self.experiment.get_images_shape()[0]*y_coords.ptp())),
                extent=limits,
                vmin=mean-2*std, vmax=mean+4*std, cmap="gray")

        # Affichage des axes.
        try:
            for axis in self.find_zone_axes():
                axe_pyplot = axis.plot_gnomonic(axe_pyplot, display=False)
        except AttributeError:
            return axe_pyplot

        # Affichage de spots.
        for spot in self:
            axe_pyplot = spot.plot_gnomonic(axe_pyplot, display=False)

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def plot_xy(self, axe_pyplot=None, *, display=True):
        """
        ** Prepare l'affichage du diagramme dans le plan du capteur. **

        Parameters
        ----------
        axe_pyplot : Axe
            Axe matplotlib qui supporte les methodes ``.scatter`` et ``.imshow``.
        display : boolean
            Si True, affiche a l'ecran en faisant appel a ``plt.show()``.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>>
        >>> diag.plot_xy(display=False)
        <AxesSubplot:title={'center':'plan camera'}, xlabel='x.Ci (pxl)', ylabel='y.Cj (pxl)'>
        >>>
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> axe = fig.add_subplot()
        >>> diag.plot_xy(axe, display=False)
        <AxesSubplot:title={'center':'plan camera'}, xlabel='x.Ci (pxl)', ylabel='y.Cj (pxl)'>
        >>>
        """
        if axe_pyplot is None:
            import matplotlib.pyplot as plt
            axe_pyplot = plt.figure().add_subplot()

        axe_pyplot.set_title("plan camera")
        axe_pyplot.set_xlabel("x.Ci (pxl)")
        axe_pyplot.set_ylabel("y.Cj (pxl)")

        # Affichage image de fond.
        try:
            image = self.get_image_xy()
        except NameError:
            pass
        else:
            mean, std = image.mean(), image.std()
            axe_pyplot.imshow(image, vmin=mean-2*std, vmax=mean+4*std, cmap="gray")

        # Affichage des spots.
        for spot in self:
            axe_pyplot = spot.plot_xy(axe_pyplot, display=False)

        if display:
            import matplotlib.pyplot as plt
            plt.show()

        return axe_pyplot

    def save_file(self, filename):
        """
        ** Enregistre un fichier contenant des informations. **

        Notes
        -----
        Les extensions prises en charge sont
        ``.dat``, ``.jpg``, ``.svg``, ``.png``

        Parameters
        ----------
        filename : str
            Nom ou chemin du fichier de destination.
            L'extension doit etre comprise dans le nom du fichier.
            Si un fichier du meme nom existe deja, il est ecrase.

        Examples
        --------
        >>> import os, tempfile
        >>> import laue
        >>>
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> rep = tempfile.mkdtemp()
        >>> diag = laue.Experiment(image)[0]
        >>> diag.save_file(os.path.join(rep, "ge_blanc.dat"))
        >>>
        """
        EXT_OK = {"dat", "jpg", "jpeg", "svg", "png"}

        assert isinstance(filename, str), \
            f"'filename' has to be a string, not a {type(filename).__name__}."
        assert "." in filename, "Le fichier doit posseder une extension."
        assert filename.split(".")[-1].lower() in EXT_OK, ("Seul les extensions "
            f"'{', '.join(EXT_OK)}' sont supportees. Pas '.{filename.split('.')[-1]}'.")

        ext = filename.split(".")[-1].lower()
        if ext == "dat":
            with open(filename, "w", encoding="utf-8") as file:
                file.write("{:<20} {:<20} {:<20}\n".format("spot_X", "spot_Y", "spot_I"))
                for spot in self:
                    file.write("{x:<20} {y:<20} {i:<20}\n".format(
                        x=spot.get_position()[0],
                        y=spot.get_position()[1],
                        i=spot.get_intensity()))
        elif ext in {"jpg", "jpeg", "svg", "png"}:
            plt = self.show(_return=True)
            plt.savefig(filename)

    def _clean(self):
        """
        ** Supprime les attributs superfux. **

        Si les spots sont modifies, cela permet de vider la memoire
        des informations desormais fausses. Si beaucoup d'images
        sont enregistrees dans la RAM, cela permet de faire de la
        place en memoire.
        """
        if os.path.exists(self.get_id()): # Il ne faut pas supprimer
            self.image_xy = None # une image que l'on ne peut pas retrouver!
        self.quality = None
        self.image_gnom = None
        self.sorted_spots = {} # Si jamais la set_calibration ou un spot change.
        self.axis = {} # Les axis de zone depandent de beaucoup de choses, on reste donc prudent.
        self.spots_set = None # On libere de la memoire en faisant ca.
        for spot in self:
            spot._clean()

    def __contains__(self, spot):
        """
        ** Verifie qu'un spot fait bien parti de ce diagramme. **

        Parameters
        ----------
        spot : laue.spot.Spot, int
            L'instance de spot dont on cherche a savoir
            si il est present ou pas. Ou bien l'index de ce spot.

        Returns
        -------
        boolean
            True si le spot est present dans ce diagramme, False sinon.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>>
        >>> len(diag)
        78
        >>> 77 in diag
        True
        >>> 78 in diag
        False
        >>>
        >>> spot = diag[0]
        >>> spot
        Spot(bbox=(1368, 1873, 6, 5), distortion=1.1804)
        >>> spot in diag
        True
        >>> 
        """
        assert isinstance(spot, (Spot, int)), ("'spot' has to be an "
            f"instance of Spot or int, not {type(spot).__name__}.")

        if isinstance(spot, int):
            if spot < 0 or spot >= len(self):
                return False
            spot = self[spot]

        if self.spots_set is None:
            self.spots_set = set(self)
        return spot in self.spots_set

    def __getitem__(self, items):
        """
        ** Permet de recuperer des elements. **

        Parameters
        ----------
        items : int, slice, tuple, laue.spot.Spot
            C'est un element qui permet de choisir un ou plusieurs
            spot dans ce diagram de laue.

        Returns
        -------
        spots : list, laue.spot.Spot
            - L'element renvoye depend du parametre d'entree ``items``:
                - int => Renvoi le item(ieme) spot. Comme si un
                ``LaueDiagram`` etait une liste de ``laue.spot.Spot``.
                    - Type renvoye: ``laue.spot.Spot``.
                - slice => Renvoi la liste des spots compris dans
                l'intervalle fournit. Comme si ``LaueDiagram`` est aussi une liste.
                    - Type renvoye: ``list``.
                - tuple => Cherche un spot ou plusieur selon une notion de distance.
                    - Le premier et/ou deuxieme argument peut etre un spot ou bien 2 nombres x et y.
                    - Le troisieme argument (factultatif) permet de preciser l'espace
                    et la metrique a utiliser pour comparer les spots a la reference.
                    C'est une chaine de caracetere expicite:
                        - "camera" (valeur par defaut si x et y de type ``int``
                        ou que le premier argument est un spot) =>
                        Distance euclidiene dans le plan de la camera (pxl).
                        - "gnomonic" (valeur par defaut si x et y de type ``float``) =>
                        Distance euclidiene dans le plan gnomonique (mm).
                        - "angle" => Cosine distance entre les vecteurs ``uq`` (axe de reflexion)
                        de chaquns des points (en degre).
                        Le premier argument est interprete comme ``2*theta`` et le second comme ``chi``.
                    - Le quatrieme argument correspond au nombre de voisins a prendre.
                    La valeur par defaut est 1. Cet argument doit etre un entier positif.
                    - Type renvoye: ``list``.
                - spot => equivalent a ``self[spot, "camera"]``.

        Raises
        ------
        IndexError
            Si aucun spot n'est quandidat ou qu'on shouaite acceder a un spot qui n'existe pas.
        ValueError
            Si les arguments fournis ne sont pas conformes.

        Examples
        -------
        >>> import numpy as np
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image, config_file="laue/examples/ge_blanc.det")[0]
        >>>
        >>> type(diag[0])
        <class 'laue.spot.Spot'>
        >>> type(diag[3:22]), len(diag[3:22])
        (<class 'list'>, 19)
        >>>
        >>> type(diag[500, 700]), len(diag[500, 700])
        (<class 'list'>, 1)
        >>> diag[500, 700].pop().get_position()
        (521.0178209801539, 724.4949372215472)
        >>>
        >>> type(diag[.2, -.3]), len(diag[.2, -.3])
        (<class 'list'>, 1)
        >>> diag[.2, -.3].pop().get_gnomonic()
        (0.2656585507727329, -0.2989256509748738)
        >>>
        >>> len(diag[0.2, .5, "gnomonic", 6])
        6
        >>> np.array([d.get_gnomonic() for d in diag[0.2, .5, "gnomonic", 3]])
        array([[0.29464949, 0.39670733],
               [0.06846284, 0.55485944],
               [0.07130572, 0.41748433]])
        >>>
        """
        if isinstance(items, (int, np.integer)):
            return self.spots[items]

        if isinstance(items, slice):
            return self.spots[items]

        if isinstance(items, tuple):
            if not items:
                raise ValueError("Vous ne pouvez pas fournir un tuple vide.")

            # Recuperation des informations.
            if isinstance(items[0], Spot):
                spot, *infos = items

                # Recherche de la metrique
                if not infos or infos[0] is None:
                    if infos:
                        _, *infos = infos
                    space = "camera"
                else:
                    space, *infos = infos

                x, y = {"camera": lambda spot: spot.get_position(),
                        "gnomonic": lambda spot: spot.get_gnomonic(),
                        "angle": lambda spot: spot.get_twicetheta_chi()
                        }.get(space, lambda spot: (None, None))[spot]
            else:
                if len(items) < 2:
                    raise ValueError("Si vous chercher a recuperer les spots"
                        "d'un voisinage, vous devez fournir au moins 2 arguments\n."
                        f"Or il y en a {len(items)}.")
                x, y, *infos = items

                # Recherche de la metrique
                if type(x) != type(y):
                    raise ValueError("Les 2 premieres coordonnees doivent etre homogenes")
                if not isinstance(x, (int, float)):
                    raise ValueError("Les 2 premiers elements doivent etre de type int ou float, "
                        f"pas de type {type(x).__name__}.")
                if not infos or infos[0] is None:
                    if infos:
                        _, *infos = infos
                    space = "camera" if isinstance(x, int) else "gnomonic"
                else:
                    space, *infos = infos

            # Recherche du nombre de voisins.
            if not infos or infos[0] is None:
                if infos:
                    _, *infos = infos
                nbr_voisins = 1
            else:
                nbr_voisins, *infos = infos

            # Verifications
            if not isinstance(nbr_voisins, int):
                raise ValueError("Le nombre de voisin doit etre un entier, "
                    f"pas {type(nbr_voisins).__name__}")
            if nbr_voisins < 1:
                raise ValueError("Le nombre de voisin doit etre strictement positif.")
            if space not in {"camera", "gnomonic", "angle"}:
                raise ValueError("Le dernier argument ne peut etre que "
                    f"'camera', 'gnomonic' ou 'angle'. Pas {repr(space)}")
            if infos:
                raise ValueError("Il y a trop de parametres.")

            # Recherche des voisins
            from laue.spot import distance

            d_list = distance((x, y), self.select_spots(), space=space)
            if nbr_voisins == 1:
                return [self.spots[np.argmin(d_list)]]
            return [self.spots[spot_ind] for spot_ind in np.argsort(d_list)[:nbr_voisins]]

        if isinstance(items, Spot):
            return self[items, None, None]

        raise ValueError("Seul les types 'int', 'slice', 'tuple' et 'Spot' sont supportees. "
            f"Or le type fourni est {type(items).__name__}.")

    def __hash__(self):
        """
        ** Fonction de hachage. **

        Permet de faire un ``dict`` ou un ``set`` avec
        des instances de ``LaueDiagram``.
        """
        return hash(self.get_id())

    def __iter__(self):
        """
        ** Permet d'iterer sur les spots. **

        Yields
        ------
        Spot
            Cede les instances des spots qui constituent
            ce diagramme dans un ordre indetermine mais invariant.
            Ces instances heritent de la classe ``laue.spot.Spot``.
            Seul les pointeurs sont cedes, ce qui implique que toute
            modification d'un spot sera globale.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> for spot in diag:
        ...     pass
        ...
        >>>
        """
        yield from self.spots

    def __len__(self):
        """
        ** Renvoi le nombre de spots. **

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = laue.Experiment(image)[0]
        >>> len(diag)
        78
        >>>
        """
        return len(self.spots)

    def __str__(self):
        """
        ** Renvoi une jolie representation du diagramme de Laue. **
        """
        return ("LaueDiagram:\n"
                f"\tname: {self.get_id()}\n"
                f"\tnbr spots: {len(self.select_spots())}\n"
                f"\tquality: {self.get_quality()}")

    def __repr__(self):
        """
        ** Renvoi une chaine evaluable de self. **
        """
        return ("LaueDiagram("
                f"name={repr(self.get_id())}, "
                f"experiment={repr(self.experiment)})")
