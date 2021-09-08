#!/usr/bin/env python3

"""
** C'est la fonction de pic search atomisee. **
-----------------------------------------------

Il est plus judicieux de l'utiliser a travers une experience
car son utilisation devient transparente et parralelisee.
"""

import cv2
import numpy as np


def atomic_pic_search(image, kernel_font, kernel_dilate, threshold):
    """
    ** Fonction 'bas niveau de pic search atomic serialisable. **

    Notes
    -----
    * Cette fonction n'est pas faite pour etre utilisee directement,
    il vaut mieux s'en servir a travers
    ``laue.experiment.base_experiment.Experiment.get_diagrams``
    ou encore via ``laue.experiment.base_experiment.Experiment.__iter__``
    car le context est mieu gere, les entrees sont plus simples et les sorties aussi.
    * Il n'y a pas de verifications sur les entrees car elles sont faite
    dans les methodes de plus haut niveau.
    * Cette fonction n'est pas parallelisee. Par contre les methodes
    de ``laue.experiment.base_experiment.Experiment`` gerent nativement le parallelisme.
    * L'utilisation de cette fonction ne fera qu'alourdir et ralentir votre code.

    Parameters
    ----------
    image : np.ndarray
        Image 2d en niveau de gris codee en np.uint16.
        C'est l'image brute, sans pre-traitement et avec le fond diffus.
    kernel_font : np.ndarray
        Le masque de l'element structurant pour l'estimation
        du fond par ouverture morphologique.
    kernel_dilate : np.ndarray
        Le masque de l'element structurant pour la dilatation morphologique
        sur l'image binarisee afin d'aglomerer les grains proches.
    threshold : float
        Le niveau de seuillage relatif a la variance de l'image.

    Returns
    -------
    list
        Une liste qui contient autant d'elements de de pic trouves.
        Les element sont des dictionaires

    Examples
    --------
    >>> import cv2
    >>> from laue import atomic_pic_search
    >>> from laue.utilities.image import read_image
    >>> image_path = "laue/examples/ge_blanc.mccd"
    >>>
    >>> image = read_image(image_path)
    >>> kernel_font = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    >>> kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    >>> threshold = 5.1
    >>>
    >>> res = atomic_pic_search(image, kernel_font, kernel_dilate, threshold)
    >>> type(res)
    <class 'list'>
    >>> len(res)
    78
    >>> res[0]["bbox"]
    (1368, 1873, 6, 5)
    >>> res[0]["distortion"]
    0.8471580534997302
    >>> res[0]["spot_im"]
    array([[  8,  10,  16,  16,   8,   5],
           [ 11,  17,  67,  76,  13,   9],
           [  7,  19, 184, 229,  14,   6],
           [  9,   6,  12,  19,   8,   4],
           [  5,   3,   3,   9,  14,   7]], dtype=uint16)
    >>> 
    """
    # Binarisation de l'image.
    bg_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_font, iterations=1)
    fg_image = image - bg_image
    thresh_image = (fg_image > threshold*fg_image.std()).astype(np.uint8)
    dilated_image = cv2.dilate(thresh_image, kernel_dilate, iterations=1)

    # Detection des contours grossiers.
    outlines, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = [cv2.boundingRect(outl) for outl in outlines]

    # Calcul des distortions.
    distortions_open = (2*np.sqrt(np.pi)) / np.array([
        cv2.arcLength(outl, True)/np.sqrt(cv2.contourArea(outl))
        for outl in outlines])

    # Preparation des arguments des spots.
    spots_args = [
        {
            "bbox": (x, y, w, h),
            "spot_im": fg_image[y:y+h, x:x+w],
            "distortion": dis,
        }
        for dis, (x, y, w, h) in zip(distortions_open, bbox)]

    return spots_args

def _pickelable_pic_search(args):
    return atomic_pic_search(*args[0]), args[1]
