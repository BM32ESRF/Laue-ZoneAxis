#!/usr/bin/env python3

"""
** Extrait l'informations de donnes pas bien rigoureuses. **
------------------------------------------------------------

Notes
-----
* Permet en particulier de faire du parsing a l'aide d'expression regulieres.
* Le module ``regex`` peut permetre d'accroitre legerement les performances.
"""

import numbers
import os
try:
	import regex as re
except ImportError:
	import re


def extract_parameters(ignore_missing=False, **detector_parameters):
    """
    ** Extrait les parametres de la camera. **

    Notes
    -----
    * Permet une grande souplesse d'utilisation.
    * Tous les parametres ne sont pas forcement a preciser.

    Parameters
    ----------
    ignore_missing : boolean, optional
        * Permet d'imposer ou non, d'avoir un retour exhaustif:
        * True => Retourne toutes les grandeurs extraites, meme si il en manque.
        * False => S'assure que tous les parametres ont etes correctement extraits.
    config_file : str
        Chemin du fichier '*.det' qui contient tous ces parametres.

    dd, detect, distance : float, int
        Plus courte distance entre l'origine du cristal et le plan de la camera.
        ||OO'|| en mm
    xcen, x0 : float, int
        Distance entre l'origine de la camera et le point d'incidence normal projetee selon X_camera.
        <O''O', Ci> en pxl
    ycen, y0 : float, int
        Distance entre l'origine de la camera et le point d'incidence normal projetee selon Y_camera.
        <O''O', Cj> en pxl
    xbet, bet, beta, angle1 : float
        Rotation du repere de la camera autour de l'axe Y_cristal dans le sens.
        axe (Y_cristal ou Ci) en degre
    xgam, gam, gamma, angle2 : float
        Rotation du repere de la camera autour de l'axe Ck qui a deja subit la rotation de ``xbet``.
        axe (OO' ou Ck) en degre
    pixelsize, size, pxlsize : float
        Dimension du cote des pixels carre du capteur.
        (taille capteur x / nbr pixels x == taille capteur y / nbr pixels y) en mm/pxl

    Returns
    -------
    dict
        Le dictionaire qui a chaque nom de parametre, associ ca valeur.
        Les clefs et les valeurs typique sont par example:
        ``{"dd": 70.0, "xcen": 1024, "ycen": 1024, "xbet": .0, "xgam": .0, "pixelsize": .080567}``

    Raises
    ------
    ValueError
        Si il y a des incoherences. (Par example si le meme parametre a 2 valeurs differentes)
    KeyError
        Si il manque des parametres.

    Examples
    --------
    >>> from laue.tools.parsing import extract_parameters
    >>> output = lambda pars: ", ".join(f"{repr(k)}: {round(pars[k], 2)}" for k in sorted(pars))
    >>>
    >>> output(extract_parameters(config_file="laue/examples/ge_blanc.det"))
    "'dd': 71.47, 'pixelsize': 0.08, 'xbet': 0.01, 'xcen': 938.51, 'xgam': -0.01, 'ycen': 1078.08"
    >>> output(extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024))
    "'dd': 70, 'pixelsize': 0.08, 'xbet': 0.0, 'xcen': 1024, 'xgam': 0.0, 'ycen': 1024"
    >>> output(extract_parameters(distance=70, angle2=.0, angle1=.0, size=.08, x0=1024, y0=1024))
    "'dd': 70, 'pixelsize': 0.08, 'xbet': 0.0, 'xcen': 1024, 'xgam': 0.0, 'ycen': 1024"
    >>>
    """
    # Verification de type et contenu.
    assert isinstance(ignore_missing, bool), \
        f"'ignore_missing' has to be a boolean, not a {type(ignore_missing).__name__}."
    if "config_file" in detector_parameters:
        assert isinstance(detector_parameters["config_file"], str), ("'file' doit etre un chemin de "
            f"fichier de type str, pas {type(detector_parameters['config_file']).__name__}.")
        assert os.path.isfile(detector_parameters["config_file"]), \
            f"{repr(detector_parameters['config_file'])} n'est pas un fichier qui existe."
    for dist in ("dd", "detect", "distance"):
        if dist in detector_parameters:
            assert isinstance(detector_parameters[dist], numbers.Number), \
                f"'{dist}' doit etre un nombre, pas un {type(detector_parameters[dist]).__name__}."
            assert detector_parameters[dist] > 0, \
                f"Toute distance doit etre positive, or elle vaut {detector_parameters[dist]}."
    for pos in ("xcen", "x0", "ycen", "y0"):
        if pos in detector_parameters:
            assert isinstance(detector_parameters[pos], numbers.Number), \
                f"'{pos}' doit etre un nombre, pas un {type(detector_parameters[pos]).__name__}."
    for angle in ("xbet", "bet", "beta", "angle1", "xgam", "gam", "gamma", "angle2"):
        if angle in detector_parameters:
            assert isinstance(detector_parameters[angle], float), \
                f"'{angle}' doit etre un flottant, pas un {type(detector_parameters[angle]).__name__}."
            assert -4.84 < detector_parameters[angle] < 4.84, ("L'angle de correction doit etre petit car "
                "un developement limite permet d'accelerer les calculs. Seulement on autorise une erreur "
                f"de 1e-4 qui correpond a 4.84 degres, pas {detector_parameters[angle]}.")
    for size in ("pixelsize", "size", "pxlsize"):
        if size in detector_parameters:
            assert isinstance(detector_parameters[size], float), \
                f"'{size}' doit etre un flottant, pas un {type(detector_parameters[size]).__name__}."
            assert detector_parameters[size] > 0, \
                f"La taille d'un pixel doit etre positive, or elle vaut {detector_parameters[size]}."

    parameters = {} # C'est le dictionaire des parametres extraits.

    # Extraction des informations du fichier.
    if "config_file" in detector_parameters:
        f_mod = re.compile(r"""(?:[+-]*
                (?:
                  \. [0-9]+ (?:_[0-9]+)*
                  (?: e [+-]? [0-9]+ (?:_[0-9]+)* )?
                | [0-9]+ (?:_[0-9]+)* \. (?: [0-9]+ (?:_[0-9]+)* )?
                  (?: e [+-]? [0-9]+ (?:_[0-9]+)* )?
                | [0-9]+ (?:_[0-9]+)*
                  e [+-]? [0-9]+ (?:_[0-9]+)*
                ))""", re.VERBOSE | re.IGNORECASE) # Model d'un flottant.
        i_mod = re.compile(r"""(?:[+-]*
                    # entier normal
                    [0-9]+ (?:_[0-9]+)*
                | 0 (?:
                    # binary
                    b [01]+ (?:_[01]+)*
                  | # octal
                    o [0-7]+ (?:_[0-7]+)*
                  | # hexadecimal
                    x [0-9a-f]+ (?:_[0-9a-f]+)*
                )
                )""", re.VERBOSE | re.IGNORECASE) # Model d'un entier.
        complete_model = re.compile(r"""^[\[\(]?\s*?(?:
                    (?P<dd>{f_mod}) [\s,]+
                    (?P<xcen>{f_mod}) [\s,]+
                    (?P<ycen>{f_mod}) [\s,]+
                    (?P<xbet>{f_mod}) [\s,]+
                    (?P<xgam>{f_mod})
                    (?: [\s,]+ (?P<pixelsize>{f_mod}))?
                )
                """.format(f_mod=f_mod.pattern, i_mod=i_mod.pattern), re.VERBOSE | re.IGNORECASE)
        dd_model = re.compile(rf"(?:dd|detect|distance)[\s:=]+(?P<dd>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        xcen_model = re.compile(rf"(?:xcen|x0)[\s:=]+(?P<xcen>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        ycen_model = re.compile(rf"(?:ycen|y0)[\s:=]+(?P<ycen>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        xbet_model = re.compile(rf"(?:xbet|bet|beta|angle1)[\s:=]+(?P<xbet>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        xgam_model = re.compile(rf"(?:xgam|gam|gamma|angle2)[\s:=]+(?P<xgam>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        pixelsize_model = re.compile(rf"(?:size|pixelsize|pxlsize)[\s:=]+(?P<pixelsize>{f_mod.pattern})",
            re.VERBOSE | re.IGNORECASE)
        keys = {"dd": dd_model, "xcen": xcen_model, "ycen": ycen_model,
                "xbet": xbet_model, "xgam": xgam_model}

        with open(detector_parameters["config_file"], "r", encoding="utf-8", errors="ignore") as file:
            for line in file: # Pour chaque ligne du fichier, on tente d'y recuperer les infos.

                # Recherche de la liste des parametres.
                complete_search = re.search(complete_model, line)
                if complete_search is not None:
                    for key in keys:
                        value = float(complete_search[key])
                        if parameters.get(key, value) != value:
                            raise ValueError(f"'{key}' value is ambigous. Is it {parameters[key]} or {value}?")
                        parameters[key] = value
                    if complete_search["pixelsize"] is not None:
                        pixelsize = float(complete_search["pixelsize"])
                        if parameters.get("pixelsize", pixelsize) != pixelsize:
                            raise ValueError("'pixelsize' value is ambigous. "
                                f"Is it {parameters['pixelsize']} or {pixelsize}?")
                        parameters["pixelsize"] = pixelsize

                # Recherches des parametres isoles.
                for key, model in {**keys, "pixelsize": pixelsize_model}.items():
                    search = re.search(model, line)
                    if search is not None:
                        value = float(search[key])
                        if parameters.get(key, value) != value:
                            raise ValueError(f"'{key}' value is ambigous. Is it {parameters[key]} or {value}?")
                        parameters[key] = value

    # 3 Extraction des informations expicites.
    for keys in [("dd", "detect", "distance"),
                 ("xcen", "x0"),
                 ("ycen", "y0"),
                 ("xbet", "bet", "beta", "angle1"),
                 ("xgam", "gam", "gamma", "angle2"),
                 ("pixelsize", "size", "pxlsize")]:
        for key in keys:
            if key in detector_parameters:
                value = detector_parameters[key]
                if parameters.get(keys[0], value) != value:
                    raise ValueError(f"'{keys[0]}' value is ambigous. Is it {parameters[keys[0]]} or {value}?")
                parameters[keys[0]] = value

    # Ajout des valeurs manquantes.
    if not ignore_missing:
        keys = {"dd", "xcen", "ycen", "xbet", "xgam", "pixelsize"}
        missing_keys = keys - set(parameters)
        if missing_keys:
            raise KeyError(f"Il manque les parametres {', '.join(missing_keys)}.")

    return parameters
