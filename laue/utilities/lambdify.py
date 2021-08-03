#!/usr/bin/env python3

"""
** Permet de faire un pont entre le calcul symbolique et matriciel. **
----------------------------------------------------------------------

Est capable de metre en forme les expressions symbolique ``sympy`` de sorte
a maximiser les performances.

Notes
-----
Si le module ``numexpr`` est installe, certaines optimisations pourront etre faites.
"""

import cloudpickle
import numpy as np
import sympy
import time

from laue.utilities.fork_lambdify import lambdify


__pdoc__ = {"cse_minimize_memory": False,
            "cse_homogeneous": False,
            "evalf": False,
            "simplify": False,
            "subs": False}


def cse_minimize_memory(r, e):
    """
    Return tuples giving ``(a, b)`` where ``a`` is a symbol and ``b`` is
    either an expression or None. The value of None is used when a
    symbol is no longer needed for subsequent expressions.

    Use of such output can reduce the memory footprint of lambdified
    expressions that contain large, repeated subexpressions.

    Examples
    --------
    >>> from sympy import cse
    >>> from laue.utilities.lambdify import cse_minimize_memory
    >>> from sympy.abc import x, y
    >>> eqs = [(x + y - 1)**2, x, x + y, (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)]
    >>> defs, rvs = cse_minimize_memory(*cse(eqs))
    >>> for i in defs:
    ...     print(i)
    ...
    (x0, x + y)
    (x1, (x0 - 1)**2)
    (x2, 2*x + 1)
    (_3, x0/x2 + x1)
    (_4, x2**x0)
    (x2, None)
    (_0, x1)
    (x1, None)
    (_2, x0)
    (x0, None)
    (_1, x)
    >>> print(rvs)
    (_0, _1, _2, _3, _4)
    >>>
    """
    if not r:
        return r, e

    from sympy import symbols

    s, p = zip(*r)
    esyms = symbols('_:%d' % len(e))
    syms = list(esyms)
    s = list(s)
    in_use = set(s)
    p = list(p)
    # sort e so those with most sub-expressions appear first
    e = [(e[i], syms[i]) for i in range(len(e))]
    e, syms = zip(*sorted(e,
        key=lambda x: -sum([p[s.index(i)].count_ops()
        for i in x[0].free_symbols & in_use])))
    syms = list(syms)
    p += e
    rv = []
    i = len(p) - 1
    while i >= 0:
        _p = p.pop()
        c = in_use & _p.free_symbols
        if c: # sorting for canonical results
            rv.extend([(s, None) for s in sorted(c, key=str)])
        if i >= len(r):
            rv.append((syms.pop(), _p))
        else:
            rv.append((s[i], _p))
        in_use -= c
        i -= 1
    rv.reverse()
    return rv, esyms

def cse_homogeneous(exprs, **kwargs):
    """
    Same as ``cse`` but the ``reduced_exprs`` are returned
    with the same type as ``exprs`` or a sympified version of the same.

    Parameters
    ----------
    exprs : an Expr, iterable of Expr or dictionary with Expr values
        the expressions in which repeated subexpressions will be identified
    kwargs : additional arguments for the ``cse`` function

    Returns
    -------
    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of sympy expressions
        The reduced expressions with all of the replacements above.

    Examples
    --------
    >>> from sympy.simplify.cse_main import cse
    >>> from sympy import cos, Tuple, Matrix
    >>> from sympy.abc import x
    >>> from laue.utilities.lambdify import cse_homogeneous
    >>> output = lambda x: type(cse_homogeneous(x)[1])
    >>> output(1)
    <class 'sympy.core.numbers.One'>
    >>> output('cos(x)')
    <class 'str'>
    >>> output(cos(x))
    cos
    >>> output(Tuple(1, x))
    <class 'sympy.core.containers.Tuple'>
    >>> output(Matrix([[1,0], [0,1]]))
    <class 'sympy.matrices.dense.MutableDenseMatrix'>
    >>> output([1, x])
    <class 'list'>
    >>> output((1, x))
    <class 'tuple'>
    >>> output({1, x})
    <class 'set'>
    """
    from sympy import cse
    if isinstance(exprs, str):
        from sympy import sympify
        replacements, reduced_exprs = cse_homogeneous(
            sympify(exprs), **kwargs)
        return replacements, repr(reduced_exprs)
    if isinstance(exprs, (list, tuple, set)):
        replacements, reduced_exprs = cse(exprs, **kwargs)
        return replacements, type(exprs)(reduced_exprs)
    if isinstance(exprs, dict):
        keys = list(exprs.keys()) # In order to guarantee the order of the elements.
        replacements, values = cse([exprs[k] for k in keys], **kwargs)
        reduced_exprs = dict(zip(keys, values))
        return replacements, reduced_exprs

    try:
        replacements, (reduced_exprs,) = cse(exprs, **kwargs)
    except TypeError: # For example 'mpf' objects
        return [], exprs
    else:
        return replacements, reduced_exprs

def evalf(x, n=15, **options):
    """
    ** Alias vers ``sympy.N``. **

    Gere recursivement les objets qui n'ont pas
    de methodes ``evalf``.
    """
    try:
        return sympy.N(x, n=n, **options)
    except AttributeError:
        if isinstance(x, (tuple, list, set)):
            return type(x)([evalf(e) for e in x])
        return type(x)(*(evalf(e) for e in x.args))

def simplify(x, **kwargs):
    """
    ** Alias vers ``sympy.simplify``. **

    Gere recursivement les objets qui n'ont pas
    de methodes pour etre directement simplifiables.

    Ajoute une factorisation recursive affin de privilegier
    les "*" plutot que les "+" de sorte a reduire les erreurs
    de calcul.
    """
    for _ in range(2):
        try:
            new_x = sympy.simplify(sympy.factor(x, deep=True, fraction=False), **kwargs)
        except AttributeError:
            if isinstance(x, (tuple, list, set)):
                new_x = type(x)([simplify(e, **kwargs) for e in x])
            else:
                new_x = type(x)(*(simplify(e, **kwargs) for e in x.args))

        if str(new_x) == str(x):
            break
        x = new_x
    return new_x

def subs(x, replacements):
    """
    ** Alias vers ``sympy.subs``. **

    Gere recursivement les objets qui n'ont pas de methode ``.subs``.
    """
    try:
        return x.subs(replacements)
    except AttributeError:
        if isinstance(x, (tuple, list, set)):
            return type(x)([subs(e, replacements) for e in x])
        return type(x)(*(subs(e, replacements) for e in x.args))

def time_cost(x):
    """
    ** Estime la complexite d'une expression. **

    Cela est tres utile pour ``sympy.simplify`` concernant
    l'argument ``measure``. Cette metrique permet de minimiser
    le temps de calcul plutot que l'elegence de l'expression.
    """
    # if hasattr(x, "__iter__"):
    #     return sum(time_cost(x_) for x_ in x)

    defs, rvs = cse_minimize_memory(*sympy.cse(x))
    return sum(sympy.count_ops(expr) for var, expr in defs)


class Lambdify:
    """
    ** Permet de manipuler plus simplement une fonction. **
    """
    def __init__(self, args, expr, *, _simplify=True):
        """
        ** Prepare la fonction. **

        Parameters
        ----------
        args : iterable
            Les parametres d'entre de la fonction.
        expr : sympy.core
            L'expresion sympy a vectoriser.
        """
        # Preparation symbolique.
        self.args = [arg for arg in sympy.sympify(args)]
        self.args_name = [str(arg) for arg in self.args]
        self.args_position = {arg: i for i, arg in enumerate(self.args_name)}
        self.expr = expr

        # Preparation vectoriele.
        self.n_expr = simplify(evalf(self.expr), measure=time_cost, inverse=True) if _simplify else self.expr
        self.fct = lambdify(self.args, self.n_expr, cse=True, modules="numpy")
        try:
            self.fct_numexpr = lambdify(self.args, self.n_expr, cse=True, modules="numexpr")
        except (TypeError, RuntimeError):
            self.fct_numexpr = None

    def __str__(self):
        """
        ** Offre une representation explicite de la fonction. **

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> print(Lambdify([x, y], cos(x + y) + x + y), end="")
        def _lambdifygenerated(x, y):
            x0 = x + y
            _0 = x0 + cos(x0)
            del x0
            return _0
        >>>
        """
        return self.fct.__doc__

    def __repr__(self):
        """
        ** Offre une representation evaluable de l'objet. **

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> Lambdify([x, y], cos(x + y) + x + y)
        Lambdify([x, y], x + y + cos(x + y))
        >>>
        """
        return f"Lambdify([{', '.join(self.args_name)}], {self.expr})"

    def __call__(self, *args, **kwargs):
        """
        ** Evalue la fonction. **

        Parameters
        ----------
        *args
            Les parametres ordonnes de la fonction.
        **kwargs
            Les parametres nomes de la fonction. Ils
            ont le dessus sur les args en cas d'ambiguite.

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>> l = Lambdify([x, y], x + y + cos(x + y))

        Les cas symboliques.
        >>> l() # Retourne l'expression sympy.
        x + y + cos(x + y)
        >>> l(x) # Complete la suite en rajoutant 'y'.
        x + y + cos(x + y)
        >>> l(y) # Complete aussi en rajoutant 'y'.
        2*y + cos(2*y)
        >>> l(x, y) # Retourne une copie de l'expression sympy.
        x + y + cos(x + y)
        >>> l(1, y=2*y) # Il est possible de faire un melange symbolique / numerique.
        2*y + cos(2*y + 1) + 1
        >>>

        Les cas purement numeriques.
        >>> import numpy as np
        >>> l(-1, 1)
        1.0
        >>> l(x=-1, y=1)
        1.0
        >>> np.round(l(0, np.linspace(-1, 1, 5)), 2)
        array([-0.46,  0.38,  1.  ,  1.38,  1.54])
        >>>
        """
        # Cas patologiques.
        if not args and not kwargs:
            return self.expr
        if len(args) > len(self.args):
            raise IndexError(f"La fonction ne prend que {len(self.args)} arguments. "
                f"Or vous en avez fournis {len(args)}.")
        
        # Recuperation des arguments complets.
        args = list(args)
        args += self.args[len(args):]
        if kwargs:
            if set(kwargs) - set(self.args_position):
                raise NameError(f"Les parametres {set(kwargs) - set(self.args_position)} "
                    f"ne sont pas admissible, seul {set(self.args_position)} sont admissibles.")
            for arg, value in kwargs.items():
                args[self.args_position[arg]] = value

        # Cas symbolique.
        if any(isinstance(a, sympy.Basic) for a in args):
            sub = {arg: value for arg, value in zip(self.args, args)}
            return subs(self.expr, sub)

        # Cas numerique.
        if (
                (self.fct_numexpr is not None)
                and (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)
                and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))
            ):
            return self.fct_numexpr(*args)
        return self.fct(*args)

    def dumps(self):
        """
        ** Transforme cet objet en binaire. **

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> l = Lambdify([x, y], cos(x + y) + x + y)
        >>> type(l.dumps())
        <class 'bytes'>
        >>>
        """
        return cloudpickle.dumps(
            {
                "args": self.args,
                "expr": self.expr,
                "n_expr": self.n_expr
            }
        )

    def loads(data):
        """
        ** Recre un objet a partir du binaire. **

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> l = Lambdify([x, y], cos(x + y) + x + y)
        >>> data = l.dumps()
        >>> type(data)
        <class 'bytes'>
        >>> type(Lambdify.loads(data))
        <class 'laue.utilities.lambdify.Lambdify'>
        >>> Lambdify.loads(data)()
        x + y + cos(x + y)
        >>>
        """
        attr = cloudpickle.loads(data)
        l = Lambdify(attr["args"], attr["n_expr"], _simplify=False)
        l.expr = attr["expr"]
        return l
