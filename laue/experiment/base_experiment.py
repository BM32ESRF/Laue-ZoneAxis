#!/usr/bin/env python3

"""
** Manipule un lot de donnees Laue. **
--------------------------------------

* Ce programe permet de faire le lien entre differents diagrammes de Laue.
* C'est le point d'entree principal.
"""

import collections
import multiprocessing
import os
import time

import cloudpickle
import cv2
import numpy as np
try:
    import psutil # Pour acceder a la memoire disponible.
except ImportError:
    psutil = None

from laue.diagram import LaueDiagram
from laue.core.geometry import transformer
from laue.spot import Spot
from laue.utilities.serialization import ExperimentPickleable
from laue.utilities.data_consistency import Recordable


__pdoc__ = {"Experiment.__getitem__": True,
            "Experiment.__iter__": True,
            "Experiment.__len__": True}


class Experiment(ExperimentPickleable, Recordable):
    """
    ** Permet de travailler sur un lot d'images. **
    """
    def __init__(self, images=(), *, config_file=None, verbose=False, **kwargs):
        """
        Parameters
        ----------
        images : str, iterable
            - Toutes les images qui constituent l'experience.
            - Peut prendre plein de forme differentes:
                - Iterable d'images. Chaque elements peut prendre 2 formes:
                    - Le chemin de l'image (str) absolu ou relatif.
                    - L'image elle meme, matrice 2d en uint16 (np.ndarray).
                        - Il vaut mieux donner le nom de l'image que l'image elle-meme
                        car la gestion de la RAM sera meilleur, surtout si il y a
                        un nombre important d'images ce qui saturerait la memoire.
                - Repertoire. Nom du dossier qui contient recursivement les images.
                - Glob expression. Par example "mon_dossier/*.tiff".
        verbose : int, optional
            * Permet d'afficher ou non des informations suplementaires.
                - 0 or False => N'affiche rien du tout, ne pollue pas l'ecran.
                - 1 or True => Affiche seulement les etapes principales.
                - 2 => Affiche les resultats intermediaires.
                - 3 => Affiche vraiment beaucoup de choses (pas tres lisible).
                - 4 => Affiche aussi des choses graphiques en plus de
                    tous ce qui est dans le terminal.

        max_space : int, optional
            Le nombre minimum de pixels qui separent 2 taches.
            Quand des taches ne sont pas separes par cet intevalle,
            elles se retrouvent aglomerees. La valeur par defaut de 5
            permet d'avoir une recherche tres hexaustive.
        threshold : float, optional
            - Seuil relatif par raport a l'ecart type de l'image sans fond.
            - Plus la valeur est grande, moins on capture de spots:
                - 3.5 => Prend enormement de spots, beaucoup de fausses detection.
                - 5.1 => Bon compromis, reste sensible sans trop capturer le fond.
                - 10 => Asser selectif, ne prend que les taches qui ressortent beaucoup.
        font_size : int, optional
            Diametre de l'element structurant qui permet d'evaluer le fond par
            une ouverture morphologique. La valeur par defaut est 21,
            normalement cette valeur est bien, il faut pas y toucher.
        ignore_errors : boolean, optional
            Permet d'ignorer certaine erreurs qui ne sont pas critiques.
            La valeur par defaut et True.
        config_file : str, optional
            Alias vers ``**detector_parameters``.
        **detector_parameters : number
            Les parametres de set_calibration de la camera deja connus.
            Il servent dans la methode ``laue.experiment.base_experiment.Experiment.set_calibration``
            a accelerer la recherche ou a la rendre plus precise.
            Pour avoir le detail sur ces parametres, voir
            ``laue.utilities.parsing.extract_parameters``.
        **bbox : number
            Ce sont les limites min et max des parametres de set_calibration a ne pas depasser.
            Les bornes minimum doivent etre precedes de '_min' et les maximum de '_max'.
            Vous pouvez par example donner ``Experiment(dd_min=90.0, dd_max=100.0)``.
            Il sont utilises dans la methode ``laue.experiment.base_experiment.Experiment.set_calibration``.
            Les noms possibles des parametres sont les meme que ``**detector_parameters``
            a l'exeption de ``pixelsize, size, pxlsize et config_file`` qui n'admettent
            pas de bornes.
            Les valeur par defaut sont:

                dd   : 60.0 mm          ; 80.0 mm
                xbet : -0.9 degre       ; 0.9 degre
                xgam : -0.9 degre       ; 0.9 degre
                xcen : milieu - 150 pxl ; milieu + 150 pxl
                ycen : milieu - 150 pxl ; milieu + 150 pxl
        **save_params
            Les parametres fournis a l'initialisateur de gestionaire d'enregistrement.
            voir ``laue.utilities.data_consistency.Recordable.__init__``.
        """
        assert isinstance(verbose, int), f"'verbose' has to be int, not {type(verbose).__name__}."
        
        max_space = kwargs.get("max_space", 5)
        assert isinstance(max_space, int), "'max_space' has to be an integer, not a %s." \
            % type(max_space).__name__
        assert max_space >= 1, f"'max_space' has to be positive. His value is '{max_space}'."
        threshold = kwargs.get("threshold", 5.1)
        assert isinstance(threshold, float), \
            f"'threshold' has to be float, not a {type(threshold).__name__}."
        assert 2.0 < threshold < 80.0, \
            f"Le seuil doit etre compris entre 2 et 50, il vaut '{threshold}'."
        font_size = kwargs.get("font_size", 21)
        assert isinstance(font_size, int), \
            f"'font_size' has to be an integer, not a {type(font_size).__name__}."
        assert font_size >= 2, ("'font_size' doit etre superieur a 1. "
            f"Il ne peut pas valoir {font_size}.")
        ignore_errors = kwargs.get("ignore_errors", True)
        assert isinstance(ignore_errors, bool), \
            f"'ignore_errors' has to be a boolean, not a {type(ignore_errors).__name__}."

        if config_file is not None:
            kwargs["config_file"] = config_file

        from laue.utilities.image import images_to_iter
        self._images = images_to_iter(images)

        self.verbose = verbose
        self.max_space = max_space
        self.threshold = threshold
        self.font_size = font_size
        self.ignore_errors = ignore_errors
        self.kwargs = kwargs

        # Precalul des constantes.
        self.kernel_font = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.font_size, self.font_size))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.max_space, self.max_space))
        self.transformer = transformer.Transformer(verbose=self.verbose) # Outil permetant de faire les transformations geometriques.
        self._predictors = {} # Predicteurs bases sur un reseau de neurones.

        # Declaration des attributs interne de memoire.
        self._len = None # Nombre de diagrames lues.
        self._buff_images = [] # La liste ordonnees des references d'images.
        self._buff_diags = [] # La liste ordonnee des diagrames lus.

        self._mean_bg = None # Fond diffus estime par la moyenne de toutes les images.
        self._shape = None # Les dimensions des matrices des images xy.
        self._calibration_parameters = None # Le dictionaire des parametres geometrique de la camera.
        self._gnomonic_matrix = None # Les matrices de transformation.

        self._images_iterator = None # Iterateur unique des informations des images.
        self._diagrams_iterator = None # Iterateur unique qui genere les diagrammes.
        self._axes_iterator = None # Iterateur unique qui cede les axes de zonne de chaque diagramme.
        self._subsets_iterator = None # Iterateur unique qui cede les bouts de grains.

        Recordable.__init__(self, **kwargs) # Instancie le gestionaire d'enregistrement.

    def set_calibration(self, *diagrams):
        """
        ** Calibration de la camera. **

        Notes
        -----
        * Ne nessecite aucune connaissances prealable sur le christal.
        * Il n'y a pas besoin d'avoir un diagramme bien calibre, il se debrouille tout seul.
        * Cette fonction peut parfois etre lente (plusieur minutes)!
        * Si vous connaissez les parametres, fournissez-les, ca ira plus vite!
        * Si cette methode a deja ete appelee une fois, elle retourne
            immeditement le resultat sans refaire les calculs.

        Parameters
        ----------
        *diagrams : optional
            Les ou le diagramme.s qui vont servir a faire la calibration.
            Si aucun diagramme n'est precise, cette methode recherche par elle meme
            les diagrammes qu'elle trouve convaincant parmis ceux qui sont disponibles.
            Il doivent etre de type ``laue.diagram.LaueDiagram``.

        Returns
        -------
        dict 
            Le dictionaire qui a chaque non de parametre, associe sa valeur numerique.
            Les clefs sont les suivantes: "dd", "xcen", "ycen", "xbet", "xgam" and "pixelsize"

        Raises
        ------
        KeyError
            Si l'utilisateur n'a pas precise les parametres vraiment indispenssables.
        ValueError
            Si il y a des incoherences dans les parametres.

        Example
        -------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> experiment = laue.experiment.base_experiment.Experiment(image, dd_min=69.5, dd_max=71.5, xbet=0.008)
        >>> parameters = experiment.set_calibration()
        >>> sorted(parameters.keys())
        ['dd', 'pixelsize', 'xbet', 'xcen', 'xgam', 'ycen']
        >>>
        """
        if self._calibration_parameters is not None: # Si on peut retourner directement,
            return self._calibration_parameters # on ne s'en prive pas.

        assert all(isinstance(diag, LaueDiagram) for diag in diagrams), \
            "Tous les diagrammes doivent etre de type 'LaueDiagram'. Or ce n'est pas le cas."

        if self.verbose:
            print("Calibration...")
            if self.verbose >= 2:
                print("    Prise en compte des parametres fournis...")

        # Constantes.
        PIXELSIZE_REF = {(2048, 2048): 0.079856, # Taille des pixels fonction de la camera.
                         (2018, 2016): 0.0734,
                         (2594, 2748): 0.031}
        PARAM_SET = {"dd", "xbet", "xgam", "xcen", "ycen"} # Les parametres non deductibles.
        PARAM_MIN = {"dd": 60.0, # Les bornes minimale par defaut.
                     "xbet": -.9,
                     "xgam": -.9,
                     "xcen": self.get_images_shape()[0]/2 - 150,
                     "ycen": self.get_images_shape()[0]/2 - 150}
        PARAM_MAX = {"dd": 80.0, # Les bornes maximales par defaut.
                     "xbet": .9,
                     "xgam": .9,
                     "xcen": self.get_images_shape()[0]/2 + 150,
                     "ycen": self.get_images_shape()[1]/2 + 150}

        # Recuperation des parametres fournis et deductibles.
        from laue.utilities.parsing import extract_parameters
        given_parameters = extract_parameters(ignore_missing=True, **self.kwargs)
        if ("pixelsize" not in given_parameters) and (self.get_images_shape() in PIXELSIZE_REF):
            given_parameters["pixelsize"] = PIXELSIZE_REF[self.get_images_shape()]
        
        elif ("pixelsize" not in given_parameters) and (self.get_images_shape() not in PIXELSIZE_REF):
            raise KeyError("Vous devez fournir le parametre 'pixelsize'.\n"
                f"Les images font {self.get_images_shape()} pxl**2. "
                f"Or, seul les 'pixelsize' des images {' et '.join(map(str, PIXELSIZE_REF))} sont connus.")

        # Recuperation des bornes.
        given_min = extract_parameters(ignore_missing=True, **{
            param[:-4]: value
            for param, value in self.kwargs.items()
            if param.endswith("_min") and len(param) > 4})
        given_max = extract_parameters(ignore_missing=True, **{
            param[:-4]: value
            for param, value in self.kwargs.items()
            if param.endswith("_max") and len(param) > 4})
        parameters_min = {param: given_min.get(param, PARAM_MIN[param]) for param in PARAM_SET}
        parameters_max = {param: given_max.get(param, PARAM_MAX[param]) for param in PARAM_SET}
        
        for param in PARAM_SET: # Verification de la coherence des bornes.
            if parameters_min[param] >= parameters_max[param]:
                raise ValueError(f"Les bornes du parametre {repr(param)} sont inversees, "
                    f"{param}_min={parameters_min[param]} et {param}_max={parameters_max[param]}.")

        # Valeur de departs des parametres.
        initial_parameters = { # Ce sont les parametres initiaux pour la descente de gradient.
            par: given_parameters.get(par,
                .5*(parameters_min[par] + parameters_max[par]))
            for par in PARAM_SET}
       
        for par, val in initial_parameters.items(): # Verification.
            if par in given_parameters and par in given_min:
                if given_parameters[par] < given_min[par]:
                    raise ValueError(f"Vous avez imposes {par}_min={given_min[par]} "
                        f"et en meme temp vous avez donnes {par}={val}!")
            if par in given_parameters and par in given_max:
                if given_parameters[par] > given_max[par]:
                    raise ValueError(f"Vous avez imposes {par}_max={given_max[par]} "
                        f"et en meme temps vous avez donnes {par}={val}!")

        # Parametres restants.
        unknown_parameters = PARAM_SET - set(given_parameters)
        if not unknown_parameters: # Si Il n'y a rien a calculer.
            self._calibration_parameters = given_parameters
            if self.verbose:
                if self.verbose >= 2:
                    print("        OK: Tout est fournis, il n'y a rien a faire.")
                print(f"    OK: Calibration terminee: {given_parameters}")
            return self._calibration_parameters

        # Extraction d'un diagramme interressant.
        if self.verbose >= 2:
            print("    Recuperation des diagrammes interressants...")
        if not diagrams: # Si l'utilisateur ne nous aide pas a trouver les bons diagrammes.
            diagrams = []
            for i, dia in enumerate(self):
                if i > 60: # On ne s'interesse qu'a la premiere minute.
                    break
                diagrams.append(dia)
            best_diagrams = [sorted(diagrams, key=lambda dia: dia.get_quality(), reverse=True).pop()]
        else: # Si l'utilisateur nous en fournit.
            best_diagrams = diagrams # C'est un tuple et non pas une liste mais c'est pas genant.
        if self.verbose >= 2:
            for dia in best_diagrams:
                print(f"        {dia.get_id()}")

        # Vectorisation des donnees pour de bonnes perfs.
        min_size = min(
            200,
            min(len(dia) for dia in best_diagrams)
            ) # Le plus petit nombre de points.
        spots_position = np.array(
            [dia.get_positions(n=min_size, sort="quality")
             for dia in best_diagrams],
            dtype=np.float32)
        spots_position = np.swapaxes(spots_position, 0, 1) # shape: (2, n_diagrams, nbr_spots)

        # Perparations des parametres pour la suite.
        vect_labels = tuple(unknown_parameters) # On recupere les nom des parametres inconus seulement.
        bounds = [(parameters_min[name], parameters_max[name]) for name in vect_labels] # Les limites des variables.
        args = (given_parameters, vect_labels, spots_position) # Les arguments en plus de la fonction de cout.
        if self.verbose >= 2:
            print(f"    calibration des parametres {vect_labels}")
            print(f"    bornes min: {tuple(b_min for b_min, _ in bounds)}")
            print(f"    bornes max: {tuple(b_max for _, b_max in bounds)}")
        
        # Recherche rapide d'un minimum par descente de gradient.
        from scipy import optimize # On ne l'importe que ici car on est pas sur de s'en servir.
        if self.verbose >= 2:
            print("    Optimsation globale, algo genetique...")
        
        if multiprocessing.current_process().name == "MainProcess" and os.cpu_count() > 4:
            attrs = ["transformer", "verbose"]
            self_bis = collections.namedtuple("PartialExperiment", attrs, defaults=[getattr(self, attr) for attr in attrs])()
            opt_res = optimize.differential_evolution(
                _Picklable(cloudpickle.dumps(self_bis), Experiment._calibration_cost,
                    {name: val for name, val in zip(("known_params", "vect_labels", "spots_position"), args)}
                    ),
                updating="deferred",
                bounds=bounds,
                disp=self.verbose >= 3, # Pour rendre la fonction verbeuse.
                polish=False, # Pour ne pas utiliser scipy.optimize.minimize a la fin.
                popsize=10, # Pour aller plus vite que la valeur de 15 par defaut.
                workers=-1) # Pour utiliser tous les cpus.
        else:
            opt_res = optimize.differential_evolution(
                self._calibration_cost,
                bounds=bounds,
                args=args,
                disp=self.verbose >= 3, # Pour rendre la fonction verbeuse.
                polish=False, # Pour ne pas utiliser scipy.optimize.minimize a la fin.
                popsize=10, # Pour aller plus vite que la valeur de 15 par defaut.
                workers=1) # Pour ne pas creer de sous processus.
                # C'est plus rapide de ne pas creer de sous processus que d'en faire... car cloudpickle est lent!
        if self.verbose >= 2:
            print(f"        Ok: cout final = {opt_res['fun']}")
        fit_parameters_vect = opt_res["x"]
        
        # Mise en forme du resultat.
        fit_parameters = {name: fit_parameters_vect[i] for i, name in enumerate(vect_labels)}
        self._calibration_parameters = {**given_parameters, **fit_parameters}
        if self.verbose:
            print(f"    OK: set_calibration terminee: {self._calibration_parameters}")
        return self._calibration_parameters

    def _calibration_cost(self, params_as_vect, known_params, vect_labels, spots_position):
        """
        ** Help for ``set_calibration``. **

        Parameters
        ----------
        params_as_vect : np.ndarray
            Le vecteur des valeurs inconues de parametres.
        :param known_params : dict
            Dictionaire des parametres connus.
        :param vect_labels : tuple
            Le tuple des nom des parametres associes au vecteur.
        spots_position : np.ndarray
            Coordonnees des points x, y des diagrammes de reference.
            shape = (2, n_diagrams, nbr_spots)
        """
        # Projection gnomonique.
        _, n_diagrams, nbr_spots = spots_position.shape
        unknown_parameters = {name: value for name, value in zip(vect_labels, params_as_vect)}
        parameters = {**unknown_parameters, **known_params}
        gnom_spots_x, gnom_spots_y = self.transformer.cam_to_gnomonic(*spots_position, parameters) # shape: (n_diagrams, nbr_pic)

        # Recherche des axes de zones intensifs.
        gnom_spots_x = ((gnom_spots_x
                        - np.repeat(gnom_spots_x.mean(axis=-1)[..., np.newaxis], nbr_spots, axis=-1)
                        ) / np.repeat(gnom_spots_x.std(axis=-1)[..., np.newaxis], nbr_spots, axis=-1))
        gnom_spots_y = ((gnom_spots_y
                        - np.repeat(gnom_spots_y.mean(axis=-1)[..., np.newaxis], nbr_spots, axis=-1)
                        ) / np.repeat(gnom_spots_y.std(axis=-1)[..., np.newaxis], nbr_spots, axis=-1))
        thetas, dists = self.transformer.hough(gnom_spots_x, gnom_spots_y) # shape: (n_diagrams, nbr_pic*(nbr_pic-1)/2)
        clusters = self.transformer.hough_reduce(thetas, dists, # shape: (n_diagrams,)
            nbr=6, tol=0.035) # On prend des qu'il y a 6 points environ allignes.

        # Calcul de l'erreur.
        projection = np.mean([np.log(1 + self.transformer.dist_line( # compris generalement entre [0.007, 0.3]
                                clusters[i][0, :], clusters[i][1, :],
                                gnom_spots_x[i, :], gnom_spots_y[i, :]).min(axis=0, initial=1.0)).mean()
                              if clusters[i].shape[-1] else 1.0 # Si il n'y a pas de droite, c'est qu'on est dans les choux.
                            for i in range(n_diagrams)]) # Moyenne des ecarts des projetes des points sur la droite la plus proche.
        scattering = np.mean([np.log(
                                    (gnom_spots_x[i, :] - gnom_spots_x[i, :].mean())**2
                                  + (gnom_spots_y[i, :] - gnom_spots_y[i, :].mean())**2
                                  + 1)
                            for i in range(n_diagrams)]) # Eparpillement [0.1, 0.95], > => bien eparpille
        cost = projection * scattering**(-4)

        if self.verbose >= 3:
            print(f"        Current parameters: {unknown_parameters}")
            print(f"        Current cost: {cost}")
            if self.verbose >= 4:
                import matplotlib.pyplot as plt
                plt.clf()
                for i in range(n_diagrams):
                    plt.subplot(n_diagrams, 3, 1 + 3*i)
                    plt.title(f"diagramme {i+1}, plan camera")
                    plt.scatter(spots_position[0, i, ...], spots_position[1, i, ...])

                    plt.subplot(n_diagrams, 3, 2 + 3*i)
                    plt.title(f"diagramme {i+1}, plan gnomonic")
                    plt.scatter(gnom_spots_x[i, :], gnom_spots_y[i, :])
                    for angle, dist in clusters[i].transpose():
                        v = np.array([np.cos(angle), np.sin(angle)])
                        u = np.array([np.sin(angle), -np.cos(angle)])
                        p = dist*v
                        plt.axline(p, p+u, lw=0.5, color="r")

                    plt.subplot(n_diagrams, 3, 3 + 3*i)
                    plt.title(f"diagramme {i+1}, hough")
                    plt.scatter(thetas[i, :], dists[i, :])
                    plt.scatter(clusters[i][0], clusters[i][1])
                plt.draw()
                plt.pause(1e-6)

        return cost

    def get_diagrams(self, *, tense_flow=False):
        """
        ** Genere les diagrammes de l'experience. **

        C'est la qu'est effectue le pic search.

        Notes
        -----
        * Performances:
            * Sur un PC (intel core i7, ssd, 8 coeurs), met environ 37 ms/diagramme.
            * Sur un PC (intel centrino, hdd, 2 coeurs), met environ 530 ms/diagramme.

        Parameters
        ----------
        tense_flow : boolean
            * True : Permet de travailler a flux tendu, c'est a dire
            de cede les diagrammes au fur a meusure qu'ils sont crees.
                * Le generateur termine quand toutes les images sont lues ou
                que le generateur d'images leve un ``StopIteration``.
                * A chaque nouvel appel de cette methode, l'iteration
                recommence a partir du debut et l'ordre reste inchange.
                * Equivalent a ``laue.experiment.base_experiment.Experiment.__iter__``.
            * False. Sinon, attend que tous les diagrammes
            soient lues afin de tout renvoyer en meme temps.

        Returns
        -------
        list
            La liste des diagrammes de type ``laue.diagram.LaueDiagram``.

        Yields
        ------
        laue.diagram.LaueDiagram
            * Chaque diagram extrait au fur a mesure qu'il arrive.
            * L'ordre est concerve pour chaque appel de cette methode a flux tendu.
            * A chaque appel on repars du debut, meme si un autre appel n'est pas termine.
            * Les sections critiques sont verouillees donc cette methode supporte le multithread.

        Examples
        --------
        >>> import laue
        >>> images = "laue/examples/*.mccd"
        >>> experiment = laue.experiment.base_experiment.Experiment(images)
        >>>
        >>> diagrams = experiment.get_diagrams()
        >>> type(diagrams)
        <class 'list'>
        >>> type(diagrams.pop())
        <class 'laue.diagram.LaueDiagram'>
        >>>
        """
        def cast_to_diagram(spots_args, name, image=None):
            """
            Met en forme du pic search pour en faire des diagrames.
            """
            laue_diagram = LaueDiagram(name, experiment=self)
            spots = [Spot(diagram=laue_diagram, identifier=i, **spot_args)
                     for i, spot_args in enumerate(spots_args)]
            laue_diagram._set_spots(spots)
            if image is not None and (
                    (not os.path.exists(name)) or (psutil is not None and psutil.virtual_memory().percent < 50)
                    ):
                laue_diagram._set_image(image)
            return laue_diagram

        def show_iterator_state(func):
            """
            Insere des commentaires.
            """
            def decorate(*func_args, **func_kwargs):
                if self.verbose:
                    print("Extraction des diagrammes...")

                start_len = len(self._buff_diags)
                for i, diag in enumerate(func(*func_args, **func_kwargs)):
                    if self.verbose >= 2:
                        print(f"    diagramme num {i+start_len} extrait: "
                              f"(...{diag.get_id()[-20:]}) "
                              f"avec {len(diag)} spots")
                    yield diag
                if self.verbose:
                    print("    OK: Tous les diagrammes sont extraits.")
            
            return decorate

        @show_iterator_state
        def _diagram_extractor(self):
            """
            Premiere vraie lecture. Cede les diagrammes.
            """
            if multiprocessing.current_process().name == "MainProcess":
                from laue.core.pic_search import _pickelable_pic_search
                from laue.utilities.multi_core import limited_imap
                with multiprocessing.Pool() as pool:
                    yield from (
                        cast_to_diagram(spots_args, name, image)
                        for spots_args, (name, image) in limited_imap(pool,
                            _pickelable_pic_search,
                            (
                                (
                                    (
                                        image,
                                        self.kernel_font,
                                        self.kernel_dilate,
                                        self.threshold
                                    ),
                                    (name, image)
                                )
                                for name, image in self.read_images(condition=(
                                    lambda im_id: not any(im_id == d.get_id() for d in self._buff_diags)
                                ))
                            )
                        )
                    )
            else:
                from laue import atomic_pic_search
                yield from (
                    cast_to_diagram(
                        atomic_pic_search(
                            image,
                            self.kernel_font,
                            self.kernel_dilate,
                            self.threshold
                        ),
                        name,
                        image
                    )
                    for name, image in self.read_images(condition=(
                        lambda im_id: not any(im_id == d.get_id() for d in self._buff_diags)
                    ))
                )
        
        if self._diagrams_iterator is None:
            self._diagrams_iterator = iter(_diagram_extractor(self))

        from laue.utilities.multi_core import RecallingIterator
        return (
            (lambda x: (yield from x))(RecallingIterator(self._diagrams_iterator, mother=self, buff_name="_buff_diags"))
            if tense_flow else list(RecallingIterator(self._diagrams_iterator, mother=self, buff_name="_buff_diags")))

    def find_subsets(self, *, tense_flow=False, **kwds):
        """
        ** Estime les grains dans chaque diagrame. **

        Sorte d'alias parallelise vers ``laue.diagram.LaueDiagram.find_subsets``.
        Le resultat est le meme que: ``[diag.find_subsets(**kwds) for diag in self]``

        Notes
        -----
        * Il est possible d'appeler plusieur fois cette methode en parallele.
        * Les sections critiques sont verouillees donc cette methode supporte le multithread.
        
        Parameters
        ----------
        tense_flow : boolean
            * True : Permet de travailler a flux tendu, c'est a dire
            de ceder les bouts de grains des diagrammes au fur a meusure qu'ils sont trouves.
                * Le generateur termine quand toutes les images sont lues ou
                que le generateur d'images leve un ``StopIteration``.
                * A chaque nouvel appel de cette methode, l'iteration
                recommence a partir du debut et l'ordre reste inchange.
                * Equivalent a ``(diag.find_subsets(**kwds) for diag in self)``.
            * False. Sinon, attend que tous les diagrammes soient lues afin de tout renvoyer en meme temps.
                * C'est equvalent a ``[diag.find_subsets(**kwds) for diag in self]``.
                * Au lieu de retourner un generateur, retourne une liste.
        **kwds
            Se sont les parametres de la fonction ``laue.diagram.LaueDiagram.find_subsets``.
            Se sont aussi ceux de la fonction ``laue.diagram.LaueDiagram.find_zone_axes``.

        Returns
        -------
        list
            Pour chaque diagramme de cette experience, cede une estimation des grains.

        Examples
        --------
        >>> import laue
        >>> images = "laue/examples/*.mccd"
        >>> experiment = laue.experiment.base_experiment.Experiment(images, config_file="laue/examples/ge_blanc.det")
        >>>
        >>> type(experiment.find_subsets())
        <class 'list'>
        >>> type(experiment.find_subsets(tense_flow=True))
        <class 'generator'>
        >>> type(next(iter(experiment.find_subsets(tense_flow=True))))
        <class 'list'>
        >>>
        """
        def show_iterator_state(func):
            """
            Insere des commentaires.
            """
            def decorate(*func_args, **func_kwargs):
                if self.verbose:
                    print("Estimation des grains...")
                
                for i, groups in enumerate(func(*func_args, **func_kwargs)):
                    if self.verbose >= 2:
                        print(f"    grain du diagramme num {i} estimes: il y a {len(groups)} clusters")
                    yield groups

                if self.verbose:
                    print("    OK: Tous les clusters de grains sont estimes.")

            return decorate

        @show_iterator_state
        def _subsets_extractor(self):
            if multiprocessing.current_process().name == "MainProcess":
                from laue.core.subsets import _jump_find_subsets
                from laue.utilities.multi_core import limited_imap
                with multiprocessing.Pool() as pool:
                    yield from (
                        (
                            diag.find_subsets(_atomic_subsets_res=args)
                            if not isinstance(args, dict) else
                            diag.find_subsets(**args)
                        )
                        for diag, args
                        in zip(
                            self,
                            limited_imap(pool,
                                _jump_find_subsets,
                                (
                                    diag.find_subsets(**kwds, _get_args=True)
                                    for _, diag in zip(self.find_zone_axes(tense_flow=True, **kwds), self)
                                )
                            )
                        )
                    )
            else:
                yield from (diag.find_subsets(**kwds) for diag in self)

        if not tense_flow:
            return list(self.find_subsets(tense_flow=True, **kwds))

        if self._subsets_iterator is None:
            self._subsets_iterator = iter(_subsets_extractor(self))

        from laue.utilities.multi_core import RecallingIterator
        return (lambda x: (yield from x))(RecallingIterator(self._subsets_iterator, mother=self))

    def find_zone_axes(self, *, tense_flow=False, **kwds):
        """
        ** Recherche l'ensemble des axes de zones. **

        Sorte d'alias parallelise vers ``laue.diagram.LaueDiagram.find_zone_axes``.
        Le resultat est le meme que: ``[diag.find_zone_axes(**kwds) for diag in self]``

        Notes
        -----
        * Il est possible d'appeler plusieur fois cette methode en parallele.
        * Les sections critiques sont verouillees donc cette methode supporte le multithread.

        Parameters
        ----------
        tense_flow : boolean
            * True : Permet de travailler a flux tendu, c'est a dire
            de ceder les axes des diagrammes au fur a meusure qu'ils sont trouves.
                * Le generateur termine quand toutes les images sont lues ou
                que le generateur d'images leve un ``StopIteration``.
                * A chaque nouvel appel de cette methode, l'iteration
                recommence a partir du debut et l'ordre reste inchange.
                * Equivalent a ``(diag.find_zone_axes(**kwds) for diag in self)``.
            * False. Sinon, attend que tous les diagrammes soient lues afin de tout renvoyer en meme temps.
                * C'est equvalent a ``[diag.find_zone_axes(**kwds) for diag in self]``.
                * Au lieu de retourner un generateur, retourne une liste.
        **kwds
            Se sont les parametres de la fonction ``laue.diagram.LaueDiagram.find_zone_axes``.

        Returns
        -------
        list
            Pour chaque diagramme de cette experience, cede la liste
            des axes de zones du diagramme. Les elements de l'ensemble
            sont de type ``laue.zone_axis.ZoneAxis``.

        Examples
        --------
        >>> import laue
        >>> images = "laue/examples/*.mccd"
        >>> experiment = laue.experiment.base_experiment.Experiment(images, config_file="laue/examples/ge_blanc.det")
        >>>
        >>> type(experiment.find_zone_axes())
        <class 'list'>
        >>> type(experiment.find_zone_axes(tense_flow=True))
        <class 'generator'>
        >>> type(next(iter(experiment.find_zone_axes(tense_flow=True))))
        <class 'list'>
        >>> type(next(iter(experiment.find_zone_axes(tense_flow=True))).pop())
        <class 'laue.zone_axis.ZoneAxis'>
        >>>
        """
        def show_iterator_state(func):
            """
            Insere des commentaires.
            """
            def decorate(*func_args, **func_kwargs):
                if self.verbose:
                    print("Extraction des axes de zone...")
                
                for i, axes in enumerate(func(*func_args, **func_kwargs)):
                    if self.verbose >= 2:
                        print(f"    axes du diagramme num {i} trouves: il y en a {len(axes)}")
                    yield axes

                if self.verbose:
                    print("    OK: Tous les axes de zone sont extraits.")

            return decorate

        @show_iterator_state
        def _axes_extractor(self):
            """
            Premiere vraie extraction.
            """
            if multiprocessing.current_process().name == "MainProcess":
                # Parallelisation des fils.
                from laue.core.zone_axes import _jump_find_zone_axes
                from laue.utilities.multi_core import limited_imap
                with multiprocessing.Pool() as pool:
                    yield from (
                        (
                            diag.find_zone_axes(_axes_args=args)
                            if not isinstance(args, dict) else
                            diag.find_zone_axes(**args)
                        )
                        for diag, args
                        in zip(
                            self,
                            limited_imap(pool,
                                _jump_find_zone_axes,
                                (
                                    diag.find_zone_axes(**kwds, _get_args=True)
                                    for diag in self
                                )
                            )
                        )
                    )

            else:
                yield from (diag.find_zone_axes(**kwds) for diag in self)

        if not tense_flow:
            return list(self.find_zone_axes(tense_flow=True, **kwds))

        if self._axes_iterator is None:
            self._axes_iterator = iter(_axes_extractor(self))

        from laue.utilities.multi_core import RecallingIterator
        return (lambda x: (yield from x))(RecallingIterator(self._axes_iterator, mother=self))

    def _get_gnomonic_matrix(self):
        """
        ** Calcul les matrices de transformation gnomonic **

        Notes
        -----
        * Permet via ``cv2`` d'avoir une image dans le plan gnomonic.
        * Les tailles des image sont le meme dans le plan
        de la camera et dans le plan gnommonic.

        Returns
        -------
        np.ndarray(np.float32) : map_x
            La premiere matrice que l'on peut voir comme une fonction
            de ``f(x_gnomon, y_gnomon) -> x_camera``, avec les coordonnees
            gnomonique exprimees en pxl.
        np.ndarray(np.float32) : map_y
            La seconde matrice ``f(x_gnomon, y_gnomon) -> y_camera``
        tuple : bornes
            Les limite en mm des pixel extremes:
            (xmin, xmax, ymin, ymax)
        """
        if self._gnomonic_matrix is not None:
            return self._gnomonic_matrix

        if self.verbose:
            print("Recuperation de la matrice gnomonic...")
        # Recherche des bornes.
        x_max, y_max = self.get_images_shape()
        xg, yg = self.transformer.cam_to_gnomonic(
            *np.meshgrid(np.arange(x_max), np.arange(y_max), copy=False),
            self.set_calibration())
        bornes = (xg.min(), xg.max(), yg.min(), yg.max())
        
        del xg, yg
        x_side = np.linspace(bornes[0], bornes[1], num=x_max)
        y_side = np.linspace(bornes[2], bornes[3], num=y_max)

        # Fonction inverse.
        map_x, map_y = self.transformer.gnomonic_to_cam(
            *np.meshgrid(x_side, y_side, copy=False),
            self.set_calibration(), dtype=np.float32)
        # map_x, map_y = map_x.astype(np.float32, copy=False), map_y.astype(np.float32, copy=False) # cv2 en a besoin.

        if self.verbose:
            print("    OK: La matrice gnomonic est calculee.")

        # Sauvegarde
        if psutil is not None and psutil.virtual_memory().percent < 75:
            self._gnomonic_matrix = (map_x, map_y, bornes)
            return self._gnomonic_matrix
        return (map_x, map_y, bornes)

    def get_mean(self):
        """
        ** Estime la moyenne des images. **

        Cela permet d'avoir une estimation du fond diffus.

        Notes
        -----
        * A cause des arrondis machine, seule les 9e15 permieres images sont considerees.
        * Ne retourne pas tant que toutes les images d'entree ne sont pas lues.

        Returns
        -------
        np.ndarray
            L'image de la moyenne des images en matrice 2d uint16.
        """
        if self._mean_bg is not None:
            return self._mean_bg

        if self.verbose:
            print("Estimation du fond par la moyenne...")

        im_gen = iter((image.astype(np.float64) for _, image in self.read_images()))
        try:
            mean_array = next(im_gen)
        except StopIteration as err:
            raise ValueError("L'experience ne contient aucune image.") from err
        for i, image in enumerate(im_gen):
            mean_array = i/(i+1) * mean_array + 1/(i+1) * image

        self._mean_bg = mean_array.astype(np.uint16)

        if self.verbose:
            print("    OK: La moyenne des images est estimee.")
        return self._mean_bg

    def get_images_shape(self):
        """
        ** Recupere les dimensions des images. **

        Returns
        -------
        tuple
            (nbr de lignes, nbr de colones), de type (int, int).
        """
        if self._shape is not None:
            return self._shape

        try:
            _, image = next(iter(self.read_images()))
        except StopIteration as err:
            raise ValueError("L'experience ne contient aucune image.") from err
        self._shape = image.shape
        return self._shape

    def read_images(self, condition=(lambda name: True)):
        """
        ** Cede le contenu des images. **

        Notes
        -----
        * Reitere depuis le debut a chaque appel.
        * Il peut y avoir plusieurs appels en parallele
        sans que cela ne genere de conflits. Dumoins tans que des threads
        ne sont pas utilises, car il n'y a pas de mecanisme de verrou.
        * A chaque appel de cette methode, l'ordre est conserve.
        * Les sections critiques sont verouillees donc cette methode supporte le multithread.

        Parameters
        ----------
        condition : callable, optional
            Une fonction de selection qui prend en entree l'identifiant de l'image
            et qui renvoi True si il faut lire l'image, sinon. Si il renvoi False,
            l'image en question est sautee.

        Yields
        ------
        name : str
            Le nom de l'image (path si possible)
        image : np.ndarray
            Le contenu de l'image

        Raises
        ------
        TypeError
            Si l'image n'est pas bien typee.
        FileNotFoundError
            Si le chemin de l'image n'est pas correcte.
        ValueError
            Si les images ne sont pas de la meme taille.

        Examples
        --------
        >>> import laue
        >>> images = "laue/examples/*.mccd"
        >>> for image in laue.experiment.base_experiment.Experiment(images):
        ...     pass
        ...
        >>>
        """
        from laue.utilities.multi_core import RecallingIterator, prevent_generator_size

        def read_and_check_any_image(image_info, image_num):
            """
            Soit retroune directement, soit lit le fichier.
            Retourne le nom de l'image et l'image elle-meme.
            Renvoi None, None si il faut ignorer cette image.
            """            
            # Mise en forme.
            if isinstance(image_info, str):
                image_name = image_info
                from laue.utilities.image import read_image
                image = read_image(image_info, ignore_errors=self.ignore_errors)
                if image is None:
                    return None, None
            elif isinstance(image_info, np.ndarray):
                image_name = f"image_{image_num}"
                image = image_info
            else:
                raise TypeError("L'image doit etre de type str ou np.array, "
                    f"pas {type(image_info).__name__}.")

            # Verifications
            if not isinstance(image, np.ndarray):
                raise TypeError(f"L'image doit etre un array numpy, pas un {type(image).__name__}.")
            if image.ndim != 2:
                raise TypeError(f"L'image {image_name} doit etre en niveau de gris pas de dimension {image.ndim}.")
            if image.dtype != np.uint16:
                raise TypeError(f"L'image {image_name} doit etre encodee en uint16, pas {image.dtype}.")
            if self._shape is None:
                self._shape = image.shape
            if self._shape != image.shape:
                raise ValueError(f"L'image {image_name} a pour taille {image.shape} tandis que les images "
                    f"precedentes ont pour taille {self._shape}. Les images ne sont pas issues de la meme experience.")

            return image_name, image

        def show_iterator_state(func):
            """
            Insere des commentaires.
            """
            def decorate(*func_args, **func_kwargs):
                if self.verbose:
                    print("Lecture des images...")
                
                for image_info in func(*func_args, **func_kwargs):
                    yield image_info
                    if self.verbose >= 2:
                        print(f"    image : {str(image_info).split('/')[-1][-40:]} cedee.")

                if self.verbose:
                    print("    OK: Toutes les images sont lues.")

            return decorate

        def update_len(func):
            """
            Tient a jour la longueur de l'experience.
            """
            def decorate(*func_args, **func_kwargs):
                for element in func(*func_args, **func_kwargs):
                    yield element
                self._len = len(self._buff_images)

            return decorate

        @update_len
        @show_iterator_state
        def _images_extractor():
            """
            Premiere vraie extraction.
            """
            yield from self._images

        @prevent_generator_size(min_size=(0 if self._buff_images else 1))
        def jump_map(multi_image_iterator):
            image_num = 0
            for image_info in multi_image_iterator:
                if condition(image_info):
                    image_name, image = read_and_check_any_image(image_info, image_num)
                else:
                    image_name = image = None
                    image_num += 1
                if image is None:
                    continue
                image_num += 1
                yield image_name, image

        if self._images_iterator is None:
            self._images_iterator = iter(_images_extractor())

        return jump_map(RecallingIterator(self._images_iterator, mother=self, buff_name="_buff_images"))

    def save_file(self, filename):
        """
        ** Enregistre un fichier contenant des informations. **

        Notes
        -----
        * Les extensions prises en charge sont ``.det``.
        * Pour les fichiers propres a chaque diagrammes, voir ``laue.diagram.LaueDiagram.save_file``.

        Parameters
        ----------
        filename : str
            Nom ou chemin du fichier de destination.
            L'extension doit etre comprise dans le nom du fichier.
            Si un fichier du meme nom existe deja, il est ecrase.

        Example
        -------
        >>> import os, tempfile
        >>> import laue
        >>>
        >>> images = ["laue/examples/ge_blanc.mccd"]
        >>> rep = tempfile.mkdtemp()
        >>> expe = laue.experiment.base_experiment.Experiment(images, dd=71.5, x0=938.5, y0=1078.1)
        >>> expe.save_file(os.path.join(rep, "fit.det"))
        """
        assert isinstance(filename, str), \
            f"'filename' has to be a string, not a {type(filename).__name__}."
        assert "." in filename, "Le fichier doit posseder une extension."
        assert filename.split(".")[-1].lower() in {"det"}, \
            f"Seule les extensions '.det' sont supportees. Pas '.{filename.split('.')[-1]}'."

        ext = filename.split(".")[-1].lower()
        if ext == "det":
            with open(filename, "w", encoding="utf-8") as file:
                file.write(
                   (f"{self.set_calibration()['dd']}, "
                    f"{self.set_calibration()['xcen']}, "
                    f"{self.set_calibration()['ycen']}, "
                    f"{self.set_calibration()['xbet']}, "
                    f"{self.set_calibration()['xgam']}, "
                    f"{self.set_calibration()['pixelsize']}, "
                    f"{self.get_images_shape()[0]}, "
                    f"{self.get_images_shape()[1]}\n"))
                file.write("Sample-Detector distance(IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n")
                file.write(f"{repr(self)}\n")
                file.write(f"Calibration done at {time.asctime()}.\n")

    def _clean(self):
        """
        ** Tente de liberer de la memoire. **

        Supprime tous les attributs qui sont suceptibles
        de prendre de la place en memoire.
        """
        if self.verbose:
            print("Suppression des attributs facultatifs...")
        self._gnomonic_matrix = None
        for diag in self:
            diag._clean()
        if self.verbose:
            print("    OK: Le volume de donnees et minimum.")

    def __getitem__(self, item):
        """
        ** Recupere un ou plusieurs diagrame.s. **

        Retroune le ou les diagrames de type ``laue.diagram.LaueDiagram``.

        Parameters
        ----------
        item
            * Ce qui permet de reconaitre un diagrame parmis tous.
                * ``int`` => Retourne le ieme diagrame, genere par la
                    ieme image lue.
                * ``slice`` => Permet de manipuler l'experience comme une
                    liste de diagrames ordones dans l'ordre de generation des images.

        Raises
        ------
        KeyError
            Si la clef est correcte mais qu'aucun diagrame ne correspond a cette clef.
        TypeError
            Si la clef n'est pas correcte.

        Examples
        --------
        >>> import laue
        >>> images = "laue/examples/*.mccd"
        >>>
        >>> type(laue.experiment.base_experiment.Experiment(images)[0])
        <class 'laue.diagram.LaueDiagram'>
        >>> type(laue.experiment.base_experiment.Experiment(images)[-1])
        <class 'laue.diagram.LaueDiagram'>
        >>>
        >>> type(laue.experiment.base_experiment.Experiment(images)[:])
        <class 'list'>
        >>> len(laue.experiment.base_experiment.Experiment(images)[:])
        2
        >>> laue.experiment.base_experiment.Experiment(images)[2:]
        []
        >>>
        """
        def get_diag_list(limit, ignore=False):
            # Cas simple ou il n'y a rien a extraire.
            if limit >= 0 and len(self._buff_diags) > limit: # Si on a deja une liste de la bone taille.
                return self._buff_diags
            if limit < 0 and len(self) and -limit <= len(self):
                return self._buff_diags
            if len(self) and ignore: # Si il faut se contenter de ce qu'on a, meme si c'est pas suffisant.
                return self._buff_diags
            if len(self) and limit >= len(self):
                raise KeyError(f"L'experience n'est faite que de {len(self)} diagrames, "
                    f"Or vous tentez d'acceder au {limit+1}eme diagrame!")
            if len(self) and -limit > len(self):
                raise KeyError(f"L'experience n'est faite que de {len(self)} diagrames, "
                    f"Or vous tentez d'acceder au rang {limit}. "
                    f"Le plus petit rang possible c'est {-len(self)}.")

            # Cas ou il faut extraire.
            if limit >= 0:
                target_limit = max(limit, 2*len(self._buff_diags))
                for i, diag in enumerate(self):
                    if i == limit:
                        break
            else:
                self.get_diagrams()

            return get_diag_list(limit=limit, ignore=ignore)

        if isinstance(item, (int, np.integer)):
            return get_diag_list(item)[item]

        if isinstance(item, slice):
            assert item.start is None or isinstance(item.start, (int, np.integer)), \
                f"Slice arguments has to be int, not {type(item.start).__name__}."
            assert item.stop is None or isinstance(item.stop, (int, np.integer)), \
                f"Slice arguments has to be int, not {type(item.stop).__name__}."
            assert item.step is None or isinstance(item.step, (int, np.integer)), \
                f"Slice arguments has to be int, not {type(item.step).__name__}."
            l1 = get_diag_list(item.start, ignore=True) if item.start is not None else []
            l2 = get_diag_list(item.stop, ignore=True) if item.stop is not None else self.get_diagrams()
            return (l1 if len(l1) > len(l2) else l2)[item]

        raise TypeError(f"La clef doit etre de type int ou slice. Pas {type(item).__name__}.")

    def __iter__(self):
        """
        ** Cede les differents diagrammes contenus dans l'experience. **

        * L'ordre est arbitraire la premiere fois mais reste le meme a chaque appel.
        * Strictement equivalent a ``self.get_diagrams(tense_flow=True)`` de
        la methode ``laue.experiment.base_experiment.Experiment.get_diagrams``.

        Yields
        ------
        diagram : laue.diagram.LaueDiagram
            Chaque diagramme contenus dans l'experience.

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> experiment = laue.experiment.base_experiment.Experiment(image)
        >>>
        >>> for diag in experiment:
        ...     pass # Allows to process each diagram as it is ready.
        ...
        >>> type(next(iter(experiment)))
        <class 'laue.diagram.LaueDiagram'>
        >>>
        """
        yield from self.get_diagrams(tense_flow=True)

    def __len__(self):
        """
        ** Nombre de diagrammes constituants l'experience. **

        Returns
        -------
        int
            Renvoi le nombre de diagrames presents dans cette experience.
            Si toutes les images ne sont pas lus, la valeur 0 est renvoyee.
        """
        if self._len is None:
            return 0
        return self._len

    def __repr__(self):
        """
        ** Renvoi une chaine evaluable de self. **
        """
        name = repr("/".join(self[0].get_id().split("/")[:-1]))
        return f"Experiment({name})"

    def __str__(self):
        """
        ** Retourne une jolie representation. **
        """
        addi_kwargs = '        \n'.join(f'{k}={v}' for k, v in self.kwargs.items())
        addi_print = f"\n    additional kwargs: \n        {addi_kwargs}" if addi_kwargs else ""
        return ("Basic Experiment:\n"
                f"    nbr reading diagrams: {len(self)}\n"
                f"    max_space: {self.max_space} pxl\n"
                f"    threshold: {self.threshold} impact/impact\n"
                f"    font_size: {self.font_size} pxl\n"
                f"    ignore_errors: {self.ignore_errors}\n"
                f"    verbose: {self.verbose}"
                f"{addi_print}")


class _Picklable:
    def __init__(self, ser_mother_class, func, kwargs):
        self.ser_mother_class = ser_mother_class
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x):
        return self.func(cloudpickle.loads(self.ser_mother_class), x, **self.kwargs)
