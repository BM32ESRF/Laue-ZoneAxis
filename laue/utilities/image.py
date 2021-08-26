#!/usr/bin/env python3

"""
** Outil de creation et de lecture d'image. **
----------------------------------------------

Permet de creer des fausse images de diagramme de laue a partir des
pics mais permet aussi de lire les image de fichiers existants.
"""

import logging
import os

import cv2
import numpy as np


def read_image(image_path, *, ignore_errors=False):
    """
    ** Lit une image sur le disque dur. **

    Parameters
    ----------
    image_path : str
        Nom de l'image, chemin absolu ou relatif, peu importe.
    ignore_errors : boolean
        Same as ``laue.experiment.base_experiment.Experiment.__init__``.
        Permet de renvoyer ``None`` plutot que de lever une exeption.

    Returns
    -------
    np.ndarray
        L'image en niveau de gris encodee en uint16.

    Raises
    ------
    FileNotFoundError
        Si l'image n'existe pas.
    ImportError
        Si il manque un module pour lire l'image.
    ValueError
        Si le fichier n'est pas lisible.

    Example
    -------
    >>> from laue.utilities.image import read_image
    >>> image = "laue/examples/ge_blanc.mccd"
    >>> read_image(image)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint16)
    >>> read_image(image).shape
    (2048, 2048)
    >>>
    """
    assert isinstance(image_path, (str, bytes)), \
        f"'image_path' has to be str or byte-like object. Not {type(image_path).__name__}."

    if not os.path.exists(image_path):
        message = f"{repr(image_path)} n'est pas un chemin existant."
        if not ignore_errors:
            raise FileNotFoundError(message)
        logging.warning(message)
        return None
    if not os.path.isfile(image_path):
        message = f"{repr(image_path)} n'est pas un fichier."
        if not ignore_errors:
            raise FileNotFoundError(message)
        logging.warning(message)
        return None
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if image is None:
        try:
            import fabio # Pour lire les fichier .mccd
        except ImportError as err:
            raise ImportError("Pour lire les fichier '.mccd' ou '.gz', "
                "il faut installer le module 'fabio'.") from err
        try:
            image = fabio.open(image_path).data
        except (KeyError, OSError) as err:
            message = f"Echec de lecture du fichier {repr(image_path)} comme une image."
            if not ignore_errors:
                raise ValueError(message) from err
            logging.warning(message)
            return None
    image = image.astype(np.uint16)
    return image

def create_image(positions, intensities=None, *, shape=None):
    """
    ** Genere syntetiquement une image de laue. **

    Paremeters
    ----------
    positions : iterable
        Les coordonnes x et y des spots en pxl dans le plan de la camera.
    intensities : iterable, optional
        L'intensite de chaque spot.
    shape : tuple, optional
        Les dimensions x, y de l'image de sortie. Par defaut, les
        valeurs maximales des positions sont utilisees.

    Returns
    -------
    image : np.ndarray
        L'image en niveau de gris codee en uint16.

    Examples
    --------
    >>> import numpy as np
    >>> from laue.utilities.image import create_image
    >>> np.random.seed(0)
    >>>
    >>> positions = np.random.uniform(0, 2048, size=(2, 600))
    >>> create_image(positions)
    array([[    0,     0,     0, ...,     0,     0,     0],
           [    0,     0,     0, ...,     0,     0,     0],
           [    0,     0,     0, ...,     0,     0,     0],
           ...,
           [    0,     0,     0, ...,  6592,  6592,  6592],
           [    0,     0,     0, ..., 12128, 12128, 12128],
           [    0,     0,     0, ..., 12128, 12128, 12128]], dtype=uint16)
    >>>
    """
    assert isinstance(positions, (np.ndarray, list, tuple)), \
        f"'positions' doit etre un iterable ordonne, pas {positions}."
    assert intensities is None or isinstance(intensities, (np.ndarray, list, tuple)), \
        f"'intensities' doit etre un iterable ordonne, pas {intensities}."
    assert shape is None or isinstance(shape, tuple), \
        f"'shape' has to be a tuple, not a {type(shape).__name__}."
    positions = np.array(positions)
    assert positions.ndim == 2, f"Les positions doivent etre une matrice 2d, pas {positions.ndim}d."
    if positions.shape[1] == 2 and positions.shape[0] != 2:
        positions = positions.transpose()
    intensities = np.ones(positions.shape[1]) if intensities is None else np.array(intensities)
    assert intensities.shape == (positions.shape[1],), ("Les positions sous-entendent qu'il y a "
        f"{positions.shape[1]} spots. Les intensites doivent donc etre de shape=({positions.shape[1]},) "
        f"et non pas {intensities.shape}.")
    shape = (int(positions[0].max()+1), int(positions[1].max()+1)) if shape is None else shape
    assert len(shape) == 2, f"L'image et en 2d, pas en {len(shape)}d."
    assert isinstance(shape[0], int) and isinstance(shape[1], int), \
        f"Les dimensions de l'image sont en pxl et donc doivent etre des entiers."

    image = np.zeros(shape=shape, dtype=np.uint16)
    intensities *= len(intensities)/intensities.mean()

    for x, y, lum in zip(*positions, intensities):
        l = int(lum) + 1
        x, y = int(round(x)), int(round(y))
        cost = int(100*lum)
        image[x-l:x+l, y-l:y+l] += cost

    return image
    
def images_to_iter(images):
    """
    ** Converti les images en un generateur d'images. **

    Parameters
    ----------
    images
        Ce qui representes les images. Que ce soit le nom
        d'un dossier, d'une image elle meme, une glob expression,
        une liste d'image ou bien un generateur.
    """
    if isinstance(images, str): # Dans le cas ou une chaine de caractere
        if os.path.isdir(images): # decrit l'ensemble des images.
            images = sorted(
                os.path.join(father, file)
                for father, _, files in os.walk(images)
                for file in files)
        else:
            images = sorted(glob.iglob(images, recursive=True))
    elif isinstance(images, (tuple, set)):
        images = list(images)

    assert hasattr(images, "__iter__"), ("'images' must to be iterable. "
        f"It can not be of type {type(images).__name__}.")

    return images
