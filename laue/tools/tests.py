#!/usr/bin/env python3

"""
** Script qui permet de faire passer des tests. **
--------------------------------------------------

Il y a 2 types de tests, les docs-tests et des tests classiques.

Les resultats des tests pousses sont ecrit dans le fichier 'tests_results.txt'
"""

# https://docs.pytest.org/en/6.2.x/reference.html

import os
import sys
import time


CALIBRATION_PARAMETERS = None


def test_read_images():
    _print("============== TEST READ IMAGES ==============")
    with CWDasRoot():
        from laue.experiment.base_experiment import Experiment
    
    for images in _find_images_dir():
        _print(images, end=" ")
        experiment = Experiment(images=images)

        t1 = time.time()
        im1 = [(name, hash(image.tobytes())) for name, image in experiment.read_images()]
        t2 = time.time()
        _print(f"{len(im1)} images: {_ftime(t2-t1)}")

        im2 = [(name, hash(image.tobytes())) for name, image in experiment.read_images()]
        assert im1 == im2

def test_find_diagrams():
    _print("============= TEST FIND DIAGRAMS =============")
    with CWDasRoot():
        from laue.experiment.base_experiment import Experiment
    
    for images in _find_images_dir():
        _print(images, end=" ")
        experiment = Experiment(images=images)

        t1 = time.time()
        di1 = [hash(diag) for diag in experiment]
        t2 = time.time()
        
        _print(f"{len(di1)} diagrams: {_ftime(t2-t1)}")

        di2 = [hash(diag) for diag in experiment]
        assert di1 == di2

def test_calibration():
    global CALIBRATION_PARAMETERS

    _print("============== TEST CALIBRATION ==============")
    with CWDasRoot():
        from laue.experiment.base_experiment import Experiment
    
    CALIBRATION_PARAMETERS = [] # Les parametres de set_calibration.

    for images in _find_images_dir():
        _print(images, end=" ")
        experiment = Experiment(images=images)
        experiment.get_diagrams() # Extraction des diagrames pour un meilleur choix.
        
        t1 = time.time()
        CALIBRATION_PARAMETERS.append(experiment.set_calibration())
        t2 = time.time()

        _print(_ftime(t2-t1))

def test_find_zone_axis():
    _print("=============== TEST ZONE AXES ===============")
    with CWDasRoot():
        from laue.experiment.base_experiment import Experiment

    for images, parameters in zip(_find_images_dir(), CALIBRATION_PARAMETERS):
        _print(images, end=" ")
        experiment = Experiment(images=images, **parameters)

        t1 = time.time()
        all_axes1 = [axes for axes in experiment.find_zone_axes()]
        t2 = time.time()

        n_axes = sum(len(axes) for axes in all_axes1)
        _print(f"{len(all_axes1)} diagrams ({n_axes} axes): {_ftime(t2-t1)}")

        all_axes2 = [axes for axes in experiment.find_zone_axes()]
        assert all_axes1 == all_axes2
        
def test_grain_separation():
    _print("=========== TEST GRAIN SEPARATION ============")
    with CWDasRoot():
        from laue.experiment.base_experiment import Experiment

    for images, parameters in zip(_find_images_dir(), CALIBRATION_PARAMETERS):
        _print(images, end=" ")
        experiment = Experiment(images=images, **parameters)

        t1 = time.time()
        all_grains1 = [diag.find_subsets() for diag in experiment]
        t2 = time.time()

        res_percent = 100*len([None for grains in all_grains1 if grains])/len(all_grains1)
        _print(f"{len(all_grains1)} diagrams: {_ftime(t2-t1)}, {res_percent:.2f}% of results.")

        all_grains2 = [diag.find_subsets() for diag in experiment]
        assert all_grains1 == all_grains2


# Help for test functions.

def _find_images_dir(images_min=10, root=os.path.expanduser("~")):
    """
    ** Recherche les images des experiences. **

    Paremeters
    ----------
    images_min : int
        Le nombre minium d'images dans un dossier.
    root : str
        Le repertoir racine a explorer recursivement.
    """
    with CWDasRoot():
        from laue.tools.multi_core import RecallingIterator
   
    def generator():
        extensions = {".mccd", ".mccd.gz", ".tiff", ".tiff.gz"}
        for father, dirs, files in os.walk(root):
            if len(files) < images_min:
                continue
            for extension in extensions:
                if len([f for f in files if f.endswith(extension)]) >= images_min:
                    yield f"{father}/*{extension}"

    if "images_iterator" not in globals():
        globals()["images_iterator"] = iter(generator())
    return RecallingIterator(globals()["images_iterator"])

def _timer(f):
    def f_bis(*args, **kwargs):
        ti = time.time()
        res = f(*args, **kwargs)
        print(f"Temps pour {f.__name__} at pid {os.getpid()}: {_ftime(time.time()-ti)} s.")
        return res
    return f_bis

def _print(*elements, **kwargs):
    if "first_print_call" not in globals():
        globals()["first_print_call"] = None
        with open("tests_results.txt", "w") as f:
            f.write(f"DATE: {time.asctime()}\n")
            f.write(f"NCPU: {os.cpu_count()}\n")
    with open("tests_results.txt", "a", encoding="utf-8") as f:
        print(*elements, **kwargs, file=f)

def _ftime(t):
    if t < 1e-6:
        return f"{t*1e9:.3f} ns"
    if t < 1e-3:
        return f"{t*1e6:.3f} us"
    if t < 1e0:
        return f"{t*1e3:.3f} ms"
    return f"{t:.3f} s"

class CWDasRoot:
    """
    ** Permet de se placer a la racine du module. **
    """
    def __init__(self):
        self.old_cwd = os.getcwd()

    def __enter__(self):
        file_path = os.path.abspath(__file__)
        root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        os.chdir(root)
        sys.path.insert(0, root)

    def __exit__(self, type, value, traceback):
        os.chdir(self.old_cwd)

if __name__ == "__main__":
    with CWDasRoot():
        os.system("clear && pytest --doctest-modules laue/ && pytest laue/tools/tests.py")
