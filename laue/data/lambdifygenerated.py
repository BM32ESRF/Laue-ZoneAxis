#!/usr/bin/env python3

"""
This code was automatically generated on Thu Aug 19 00:10:29 2021.
"""

import sympy
import numpy as np

HASH = '61fd130d13f54ec87cb53ade5fa2d667'

def cam_to_gnomonic(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 8, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _cam_to_gnomonic_sympy
        return _cam_to_gnomonic_sympy()
    args = list(args)
    if len(args) < 8:
        args += sympy.symbols(' '.join(['x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x_cam': 0, 'y_cam': 1, 'dd': 2, 'xcen': 3, 'ycen': 4, 'beta': 5, 'gamma': 6, 'pixelsize': 7}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _cam_to_gnomonic_sympy
        return subs(_cam_to_gnomonic_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _cam_to_gnomonic_numpy128
        return _cam_to_gnomonic_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _cam_to_gnomonic_numexpr
        return _cam_to_gnomonic_numexpr(*args)
    from numpy_lambdify import _cam_to_gnomonic_numpy
    return _cam_to_gnomonic_numpy(*args)

def gnomonic_to_cam(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 8, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x_gnom', 'y_gnom', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _gnomonic_to_cam_sympy
        return _gnomonic_to_cam_sympy()
    args = list(args)
    if len(args) < 8:
        args += sympy.symbols(' '.join(['x_gnom', 'y_gnom', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x_gnom': 0, 'y_gnom': 1, 'dd': 2, 'xcen': 3, 'ycen': 4, 'beta': 5, 'gamma': 6, 'pixelsize': 7}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x_gnom', 'y_gnom', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _gnomonic_to_cam_sympy
        return subs(_gnomonic_to_cam_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _gnomonic_to_cam_numpy128
        return _gnomonic_to_cam_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _gnomonic_to_cam_numexpr
        return _gnomonic_to_cam_numexpr(*args)
    from numpy_lambdify import _gnomonic_to_cam_numpy
    return _gnomonic_to_cam_numpy(*args)

def cam_to_thetachi(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 8, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _cam_to_thetachi_sympy
        return _cam_to_thetachi_sympy()
    args = list(args)
    if len(args) < 8:
        args += sympy.symbols(' '.join(['x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x_cam': 0, 'y_cam': 1, 'dd': 2, 'xcen': 3, 'ycen': 4, 'beta': 5, 'gamma': 6, 'pixelsize': 7}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x_cam', 'y_cam', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _cam_to_thetachi_sympy
        return subs(_cam_to_thetachi_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _cam_to_thetachi_numpy128
        return _cam_to_thetachi_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _cam_to_thetachi_numexpr
        return _cam_to_thetachi_numexpr(*args)
    from numpy_lambdify import _cam_to_thetachi_numpy
    return _cam_to_thetachi_numpy(*args)

def thetachi_to_cam(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 8, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'theta', 'chi', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _thetachi_to_cam_sympy
        return _thetachi_to_cam_sympy()
    args = list(args)
    if len(args) < 8:
        args += sympy.symbols(' '.join(['theta', 'chi', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'theta': 0, 'chi': 1, 'dd': 2, 'xcen': 3, 'ycen': 4, 'beta': 5, 'gamma': 6, 'pixelsize': 7}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['theta', 'chi', 'dd', 'xcen', 'ycen', 'beta', 'gamma', 'pixelsize'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _thetachi_to_cam_sympy
        return subs(_thetachi_to_cam_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _thetachi_to_cam_numpy128
        return _thetachi_to_cam_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _thetachi_to_cam_numexpr
        return _thetachi_to_cam_numexpr(*args)
    from numpy_lambdify import _thetachi_to_cam_numpy
    return _thetachi_to_cam_numpy(*args)

def thetachi_to_gnomonic(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 2, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'theta', 'chi'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _thetachi_to_gnomonic_sympy
        return _thetachi_to_gnomonic_sympy()
    args = list(args)
    if len(args) < 2:
        args += sympy.symbols(' '.join(['theta', 'chi'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'theta': 0, 'chi': 1}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['theta', 'chi'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _thetachi_to_gnomonic_sympy
        return subs(_thetachi_to_gnomonic_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _thetachi_to_gnomonic_numpy128
        return _thetachi_to_gnomonic_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _thetachi_to_gnomonic_numexpr
        return _thetachi_to_gnomonic_numexpr(*args)
    from numpy_lambdify import _thetachi_to_gnomonic_numpy
    return _thetachi_to_gnomonic_numpy(*args)

def gnomonic_to_thetachi(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 2, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x_gnom', 'y_gnom'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _gnomonic_to_thetachi_sympy
        return _gnomonic_to_thetachi_sympy()
    args = list(args)
    if len(args) < 2:
        args += sympy.symbols(' '.join(['x_gnom', 'y_gnom'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x_gnom': 0, 'y_gnom': 1}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x_gnom', 'y_gnom'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _gnomonic_to_thetachi_sympy
        return subs(_gnomonic_to_thetachi_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _gnomonic_to_thetachi_numpy128
        return _gnomonic_to_thetachi_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _gnomonic_to_thetachi_numexpr
        return _gnomonic_to_thetachi_numexpr(*args)
    from numpy_lambdify import _gnomonic_to_thetachi_numpy
    return _gnomonic_to_thetachi_numpy(*args)

def dist_cosine(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 4, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'theta_1', 'chi_1', 'theta_2', 'chi_2'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _dist_cosine_sympy
        return _dist_cosine_sympy()
    args = list(args)
    if len(args) < 4:
        args += sympy.symbols(' '.join(['theta_1', 'chi_1', 'theta_2', 'chi_2'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'theta_1': 0, 'chi_1': 1, 'theta_2': 2, 'chi_2': 3}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['theta_1', 'chi_1', 'theta_2', 'chi_2'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _dist_cosine_sympy
        return subs(_dist_cosine_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _dist_cosine_numpy128
        return _dist_cosine_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _dist_cosine_numexpr
        return _dist_cosine_numexpr(*args)
    from numpy_lambdify import _dist_cosine_numpy
    return _dist_cosine_numpy(*args)

def dist_euclidian(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 4, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x1', 'y1', 'x2', 'y2'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _dist_euclidian_sympy
        return _dist_euclidian_sympy()
    args = list(args)
    if len(args) < 4:
        args += sympy.symbols(' '.join(['x1', 'y1', 'x2', 'y2'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x1': 0, 'y1': 1, 'x2': 2, 'y2': 3}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x1', 'y1', 'x2', 'y2'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _dist_euclidian_sympy
        return subs(_dist_euclidian_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _dist_euclidian_numpy128
        return _dist_euclidian_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _dist_euclidian_numexpr
        return _dist_euclidian_numexpr(*args)
    from numpy_lambdify import _dist_euclidian_numpy
    return _dist_euclidian_numpy(*args)

def dist_line(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 4, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'phi', 'mu', 'x', 'y'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _dist_line_sympy
        return _dist_line_sympy()
    args = list(args)
    if len(args) < 4:
        args += sympy.symbols(' '.join(['phi', 'mu', 'x', 'y'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'phi': 0, 'mu': 1, 'x': 2, 'y': 3}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['phi', 'mu', 'x', 'y'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _dist_line_sympy
        return subs(_dist_line_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _dist_line_numpy128
        return _dist_line_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _dist_line_numexpr
        return _dist_line_numexpr(*args)
    from numpy_lambdify import _dist_line_numpy
    return _dist_line_numpy(*args)

def hough(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 4, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'x_a', 'y_a', 'x_b', 'y_b'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _hough_sympy
        return _hough_sympy()
    args = list(args)
    if len(args) < 4:
        args += sympy.symbols(' '.join(['x_a', 'y_a', 'x_b', 'y_b'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'x_a': 0, 'y_a': 1, 'x_b': 2, 'y_b': 3}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['x_a', 'y_a', 'x_b', 'y_b'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _hough_sympy
        return subs(_hough_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _hough_numpy128
        return _hough_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _hough_numexpr
        return _hough_numexpr(*args)
    from numpy_lambdify import _hough_numpy
    return _hough_numpy(*args)

def inter_line(*args, **kwargs):
    """
    ** Choose the most suitable function according to
    the type and size of the input data. **

    Parameters
    ----------
    *args
        Les parametres ordonnes de la fonction.
    **kwargs
        Les parametres nomes de la fonction. Ils
        ont le dessus sur les args en cas d'ambiguite.
    """
    assert len(args) <= 4, f'The function cannot take {len(args)} arguments.'
    assert not set(kwargs) - {'phi_1', 'mu_1', 'phi_2', 'mu_2'}, f'You cannot provide {kwargs}.'
    if not args and not kwargs:
        from sympy_lambdify import _inter_line_sympy
        return _inter_line_sympy()
    args = list(args)
    if len(args) < 4:
        args += sympy.symbols(' '.join(['phi_1', 'mu_1', 'phi_2', 'mu_2'][len(args):]))
    if kwargs:
        for arg, value in kwargs.items():
            args[{'phi_1': 0, 'mu_1': 1, 'phi_2': 2, 'mu_2': 3}[arg]] = value
    if any(isinstance(a, sympy.Basic) for a in args):
        sub = {arg: value for arg, value in zip(['phi_1', 'mu_1', 'phi_2', 'mu_2'], args)}
        from laue.utilities.lambdify import subs
        from sympy_lambdify import _inter_line_sympy
        return subs(_inter_line_sympy(), sub)
    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):
        from numpy128_lambdify import _inter_line_numpy128
        return _inter_line_numpy128(*args)
    if (
            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
        ):
        from numexpr_lambdify import _inter_line_numexpr
        return _inter_line_numexpr(*args)
    from numpy_lambdify import _inter_line_numpy
    return _inter_line_numpy(*args)

