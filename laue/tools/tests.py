#!/usr/bin/env python3

"""
** Script qui permet de faire passer des tests sur des donnees reeles. **
-------------------------------------------------------------------------

Les resultats des tests pousses sont ecrit dans le fichier 'tests_results.txt'
"""

# https://docs.pytest.org/en/6.2.x/reference.html

import itertools
import os
import sys
import time

import numpy as np


CALIBRATION_PARAMETERS = None

# Tests theoriques.

def test_geometry_dtype():
    """
    S'assure que les types soient bien concerves aux cours
    des operations de transformation geometriques.
    """
    _print("============ TEST GEOMETRY DTYPE =============")
    with CWDasRoot():
        from laue.geometry import Transformer
        from laue.tools.parsing import extract_parameters
    parameters = extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024)
    trans = Transformer()

    _print("cam_to_gnomonic:")
    for dtype in [np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)]:
        for boucle in range(3):
            rtype = trans.cam_to_gnomonic(np.zeros(1), np.zeros(1), parameters, dtype=dtype).dtype.type
            _print(f"\tboucle {boucle}: {dtype.__name__}->{rtype.__name__}")
            assert rtype == dtype

    _print("gnomonic_to_cam:")
    for dtype in [np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)]:
        for boucle in range(3):
            rtype = trans.gnomonic_to_cam(np.zeros(1), np.zeros(1), parameters, dtype=dtype).dtype.type
            _print(f"\tboucle {boucle}: {dtype.__name__}->{rtype.__name__}")
            assert rtype == dtype

    _print("dist_line:")
    for dtype in [np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)]:
        rtype = trans.dist_line(
            np.array([0, np.pi/2]), np.array([1, 1]),
            np.array([0, 1, 3, 0]), np.array([0, 1, 3, 1]),
            dtype=dtype).dtype.type
        _print(f"\t{dtype.__name__}->{rtype.__name__}")
        assert rtype == dtype

    _print("hough:")
    for dtype in [np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)]:
        rtype = trans.hough(
            np.array([-1, 0, 1]), np.array([-1, 0, 1]),
            dtype=dtype).dtype.type
        _print(f"\t{dtype.__name__}->{rtype.__name__}")
        assert rtype == dtype

    _print("hough_reduce:")
    for dtype in [np.float16, np.float32, np.float64]:
        rtype = trans.hough_reduce(
            *trans.hough(
                np.array([-1, 0, 1]), np.array([-1, 0, 1]),
                dtype=dtype
            ),
            dtype=dtype
        ).dtype.type
        _print(f"\t{dtype.__name__}->{rtype.__name__}")
        assert rtype == dtype

    _print("inter_lines:")
    for dtype in [np.float16, np.float32, np.float64, (getattr(np, "float128") if hasattr(np, "float128") else np.float64)]:
        rtype = trans.inter_lines(
            np.array([0, np.pi/2]), np.array([1, 1]),
            dtype=dtype).dtype.type
        _print(f"\t{dtype.__name__}->{rtype.__name__}")
        assert rtype == dtype

def test_geometry_shape():
    """
    S'assure que les dimensions soient concervees..
    """
    _print("============ TEST GEOMETRY SHAPE =============")
    with CWDasRoot():
        from laue.geometry import Transformer
        from laue.tools.parsing import extract_parameters
    parameters = extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024)
    transformer = Transformer()

    shape = tuple(np.random.randint(1, 10, size=np.random.randint(1, 5)))
    _print(f"shape: {shape}")

    for boucle in range(3): # On fait 2 boucle car ca recompile en cours de route.
        for func in [transformer.cam_to_gnomonic, transformer.gnomonic_to_cam]:
            res = func(np.zeros(shape=shape), np.zeros(shape=shape), parameters)
            _print(f"boucle {boucle}, {func.__name__}(...).shape -> {res.shape}")
            assert res.shape == (2,) + shape

    res = transformer.dist_line(
        np.ones(shape=shape), np.ones(shape=shape),
        np.zeros(shape=shape), np.zeros(shape=shape))
    _print(f"dist_line(...).shape -> {res.shape}")
    assert res.shape == (*shape, *shape)

    res = transformer.hough(
        np.random.normal(size=shape), np.random.normal(size=shape))
    _print(f"hough(...).shape -> {res.shape}")
    assert res.shape == (2,) + shape[:-1] + ((shape[-1]*(shape[-1]-1))//2,)

    res = transformer.inter_lines(
        np.random.uniform(-np.pi, np.pi, size=shape),
        np.random.uniform(0, 2, size=shape))
    _print(f"inter_lines(...).shape -> {res.shape}")
    assert res.shape == (2,) + shape[:-1] + ((shape[-1]*(shape[-1]-1))//2,)

def test_geometry_bij():
    """
    S'assure qu'il y ai bien une bijection entre
    les equations de passage du plan gnomonic
    a celui de la camera et inversement.
    """
    _print("=============== TEST BIJECTION ===============")
    import sympy
    from sympy.vector import CoordSys3D
    with CWDasRoot():
        from laue.geometry import Transformer

    # Declaration des variables.
    N = CoordSys3D('N')
    transformer = Transformer()
    xg, yg = sympy.symbols("xg yg", real=True)
    xc, yc = sympy.symbols("xc yc", real=True)
    uf_x, uf_y, uf_z = sympy.symbols("uf_x uf_y uf_z", real=True)
    u_f = uf_x*N.i + uf_y*N.j + uf_z*N.k
    uq_x, uq_y, uq_z = sympy.symbols("uq_x uq_y uq_z", real=True)
    u_q = uq_x*N.i + uq_y*N.j + uq_z*N.k

    def is_zero(expr):
        """
        S'assure que l'expression vaille 0.
        """
        expr = sympy.simplify(expr)
        if expr == 0:
            _print("ok formel")
            return True
        _print("fail formel, ", end="")

        symbols = list(expr.free_symbols)
        x_vect = np.linspace(-1, 1, 21)
        fct = sympy.lambdify(symbols, expr, modules="numpy")
        args = np.meshgrid(*((x_vect,)*len(symbols)), copy=False)
        res = fct(*args)
        is_ok = np.nanmax(np.abs(res)) < 1e-8
        if not is_ok:
            _print("failed numerical")
            _print(f"\texpr = {expr}")
            _print(f"\tmax(abs(val)) = {np.nanmax(np.abs(res))}")
            return False
        _print("ok numerical")
        return True


    # Test des bijections des courtes etapes.

    ## uf <=> camera
    _print("uf2cam o cam2uf: ", end="")
    xc_bis, yc_bis = transformer.get_expr_uf_to_cam(
        *transformer.get_expr_cam_to_uf(xc, yc))
    assert is_zero((xc_bis - xc)**2 + (yc_bis - yc)**2)

    _print("cam2uf o uf2cam: ", end="")
    uf_x_bis, uf_y_bis, uf_z_bis = transformer.get_expr_cam_to_uf(
        *transformer.get_expr_uf_to_cam(uf_x, uf_y, uf_z))
    u_f_bis = uf_x_bis*N.i + uf_y_bis*N.j + uf_z_bis*N.k
    assert is_zero((u_f ^ u_f_bis).magnitude())

    ## uq <=> uf

    _print("uq2uf o uf2uq: ", end="")
    uf_x_bis, uf_y_bis, uf_z_bis = transformer.get_expr_uq_to_uf(
        *transformer.get_expr_uf_to_uq(uf_x, uf_y, uf_z))
    u_f_bis = uf_x_bis*N.i + uf_y_bis*N.j + uf_z_bis*N.k
    assert is_zero((u_f ^ u_f_bis).magnitude())

    _print("uf2uq o uq2uf: ", end="")
    uq_x_bis, uq_y_bis, uq_z_bis = transformer.get_expr_uf_to_uq(
        *transformer.get_expr_uq_to_uf(uq_x, uq_y, uq_z))
    u_q_bis = uq_x_bis*N.i + uq_y_bis*N.j + uq_z_bis*N.k
    assert is_zero((u_q ^ u_q_bis).magnitude())

    ## uq <=> gnomonic

    _print("uq2gnom o gnom2uq: ", end="")
    xg_bis, yg_bis = transformer.get_expr_uq_to_gnomonic(
        *transformer.get_expr_gnomonic_to_uq(xg, yg))
    assert is_zero((xg_bis - xg)**2 + (yg_bis - yg)**2)

    _print("gnom2uq o uq2gnom: ", end="")
    uq_x_bis, uq_y_bis, uq_z_bis = transformer.get_expr_gnomonic_to_uq(
        *transformer.get_expr_uq_to_gnomonic(uq_x, uq_y, uq_z))
    u_q_bis = uq_x_bis*N.i + uq_y_bis*N.j + uq_z_bis*N.k
    assert is_zero((u_q ^ u_q_bis).magnitude())

    # Test des grandes etapes.

    ## camera <=> gnomonic

    cam_to_gnom = transformer.get_fct_cam_to_gnomonic()
    gnom_to_cam = transformer.get_fct_gnomonic_to_cam()
    camgnom_rond_gnomcam = cam_to_gnom(*gnom_to_cam(xg, yg))
    gnomcam_rond_camgnom = gnom_to_cam(*cam_to_gnom(xc, yc))

    _print("cam2gnom o gnom2cam: ", end="")
    assert is_zero((camgnom_rond_gnomcam[0] - xg)**2 + (camgnom_rond_gnomcam[1] - yg)**2)
    _print("gnom2cam o cam2gnom: ", end="")
    assert is_zero((gnomcam_rond_camgnom[0] - xg)**2 + (gnomcam_rond_camgnom[1] - yc)**2)


# Tests sur les donnes reelles.

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
        all_axes1 = experiment.find_zone_axes(tense_flow=False)
        t2 = time.time()

        n_axes = sum(len(axes) for axes in all_axes1)
        _print(f"{len(all_axes1)} diagrams ({n_axes} axes): {_ftime(t2-t1)}")

        all_axes2 = experiment.find_zone_axes(tense_flow=False)
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
        Le repertoire racine a explorer recursivement.
    """
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
