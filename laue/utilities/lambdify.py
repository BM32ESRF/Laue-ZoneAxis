#!/usr/bin/env python3

"""
** Permet de faire un pont entre le calcul symbolique et matriciel. **
----------------------------------------------------------------------

Est capable de metre en forme les expressions symbolique ``sympy`` de sorte
a maximiser les performances.

Notes
-----
Si le module ``numexpr`` est installe, certaines optimisations pouront etre faites.
"""

import itertools
import multiprocessing
import numbers
import os
import pickle
import timeit

import numpy as np
import sympy

import laue
from laue.utilities.fork_lambdify import lambdify
from laue.utilities.multi_core import NestablePool as Pool


__pdoc__ = {"Lambdify.__getstate__": True,
            "Lambdify.__setstate__": True,
            "Lambdify.__call__": True,
            "Lambdify.__str__": True,
            "evalf": False,
            "subs": False}


def _generalize(f):
    def f_bis(x, *args, **kwargs):
        try:
            return f(x, *args, **kwargs)
        except AttributeError:
            if isinstance(x, (tuple, list, set)):
                return type(x)([f(e, *args, **kwargs) for e in x])
            else:
                return type(x)(*(f(e, *args, **kwargs) for e in x.args))
    return f_bis

@_generalize
def _sub_float_rat(x):
    """Remplace certain flotants par des rationels."""
    repl = {f: int(round(f)) for f in x.atoms(sympy.Float) if round(f) == round(f, 5)}
    x = subs(x, repl)
    repl = True
    while repl:
        cand_pow = [p for p in x.atoms(sympy.Pow) if isinstance(p.exp, sympy.Float)]
        cand_rat = [sympy.Rational(p.exp).limit_denominator(10) for p in cand_pow]
        repl = {p: sympy.Pow(p.base, r) for p, r in zip(cand_pow, cand_rat) if round(p.exp, 5) == round(r, 5)}
        x = subs(x, repl)
    return x

def _branch_simplify(x, measure):
    """Simplifie une simple expression sympy."""
    def main_(x, measure):
        cand = {x, x.rewrite(sympy.Piecewise)}
        new_cand = set()
        for _ in range(4):
            with Pool() as pool:
                proc = []
                for x_ in cand:
                    proc.append(pool.apply_async(sympy.cancel, args=(x_,)))
                    proc.append(pool.apply_async(sympy.factor, args=(x_,), kwds={"deep":True, "fraction":False}))
                    proc.append(pool.apply_async(sympy.factor, args=(x_,), kwds={"deep":True, "fraction":False, "gaussian":True}))
                    proc.append(pool.apply_async(sympy.logcombine, args=(x_,), kwds={"force":True}))
                    proc.append(pool.apply_async(sympy.powsimp, args=(x_,), kwds={"deep":True, "force":True, "measure":measure}))
                    proc.append(pool.apply_async(sympy.radsimp, args=(x_,), kwds={"max_terms":10}))
                    proc.append(pool.apply_async(sympy.ratsimp, args=(x_,)))
                    proc.append(pool.apply_async(sympy.sqrtdenest, args=(x_,), kwds={"max_iter":10}))
                    proc.append(pool.apply_async(sympy.simplify, args=(x_,), kwds={"measure":measure, "inverse":True, "rational":False}))
                    proc.append(pool.apply_async(sympy.trigsimp, args=(x_,), kwds={"method":"combined"}))
                    proc.append(pool.apply_async(sympy.trigsimp, args=(x_,), kwds={"method":"fu"}))
                    proc.append(pool.apply_async(sympy.trigsimp, args=(x_,), kwds={"method":"groebner"}))
                    proc.append(pool.apply_async(sympy.trigsimp, args=(x_,), kwds={"method":"matching"}))
                for p in proc:
                    try:
                        new_cand.add(p.get(timeout=10))
                    except multiprocessing.TimeoutError:
                        pass
                    except (ValueError, TypeError, AttributeError, sympy.CoercionFailed, sympy.PolynomialError):
                        pass

            new_cand = {_select(*(new_cand | cand), measure=measure)}
            if new_cand == cand:
                break
            cand = new_cand
        return new_cand.pop()

    return _select(_generalize(main_)(x, measure), x, measure=measure)

def _select(*xs, measure):
    """Selectione la meilleur expression."""
    if multiprocessing.current_process().name == "MainProcess":
        with Pool() as pool:
            costs = pool.map(measure, xs)
    else:
        costs = [measure(e) for e in xs]
    return xs[np.argmin(costs)]

def _cse_simp(x, measure):
    """Simplifi chaque paquets sous expression de cse"""
    def build(defs, rvs):
        for var, e in defs[::-1]:
            rvs = subs(rvs, {var: e})
        return rvs

    changed = True
    while changed:
        changed = False
        defs, rvs = cse_homogeneous(x)

        with Pool() as pool:
            proc = []
            for _, e in defs:
                proc.append(pool.apply_async(_branch_simplify, args=(e,), kwds={"measure": measure}))
            proc.append(pool.apply_async(_branch_simplify, args=(rvs,), kwds={"measure": measure}))

            for i, (var, e) in enumerate(defs.copy()):
                e_ = proc[i].get()
                if e_ != e:
                    changed = True
                    defs[i] = (var, e_)
            rvs_ = proc[-1].get()
            if rvs_ != rvs:
                changed = True
                rvs = rvs_

        if changed:
            x = build(defs, rvs)
    return x


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

def evalf(x, n=15):
    """
    ** Alias vers ``sympy.N``. **

    Gere recursivement les objets qui n'ont pas
    de methodes ``evalf``.
    """
    def basic_evalf(x, n):
        """Alias recursif ver evalf natif de sympy. int -> float"""
        if isinstance(x, (sympy.Atom, numbers.Number)):
            if str(x) == "pi":
                return sympy.pi.evalf(n=n)
            if str(x) == "E":
                return sympy.E.evalf(n=n)
            return x
        try:
            x = type(x)(*(basic_evalf(e, n=n) for e in x.args))
        except AttributeError:
            pass
        try:
            return sympy.N(x, n=n)
        except AttributeError:
            return x

    if isinstance(x, (tuple, list, set)):
        return type(x)([evalf(e, n=n) for e in x])

    x = basic_evalf(x, n=n)
    x = _sub_float_rat(x)
    return x

def simplify(x, measure, verbose=False):
    """
    ** Triture l'expression pour minimiser le temps de calcul. **
    """
    if verbose:
        print(f"simplify: {cse_homogeneous(x)}...")
        print(f"\tbegin cost: {measure(x)}")

    x = _sub_float_rat(x)
    if verbose >= 2:
        print(f"\tafter float to rational: {measure(x)}")
    x = _cse_simp(x, measure=measure)
    if verbose >= 2:
        print(f"\tafter cse_simp: {measure(x)}")
    x = _branch_simplify(x, measure=measure)
    if verbose >= 2:
        print(f"\tafter global simp: {measure(x)}")
    x = evalf(x, n=35)

    if verbose:
        print(f"\tfinal cost: {measure(x)}")
        print(f"\tfinal expr: {cse_homogeneous(x)}...")

    return x

@_generalize
def subs(x, replacements):
    """
    ** Alias vers ``sympy.subs``. **

    Gere recursivement les objets qui n'ont pas de methode ``.subs``.
    """
    return x.subs(replacements)


class TimeCost:
    """
    ** Estime le cout d'une expression. **
    """
    def __init__(self):
        self.costs = {}

        self._zero = np.zeros(shape=(1000, 1000))
        self._one = np.ones(shape=(1000, 1000))
        self._false = self._zero.astype(bool)
        self._true = self._one.astype(bool)
        self._negone = -self._one
        self._two = 2.0*self._one
        self._bat = .32267452*self._one
        self._comp = (1.0 + 1.0j)*self._one

        self._tests = {
            "Abs": (lambda: np.abs(self._negone)),
            "Add": (lambda: self._bat + self._bat),
            "And": (lambda: self._true & self._true),
            "Equality": (lambda: self._bat == self._bat),
            "GreaterThan": (lambda: self._one >= self._one),
            "LessThan": (lambda: self._one <= self._one),
            "MatAdd": (lambda: self._bat + self._bat),
            "MatMul": (lambda: self._bat @ self._bat),
            "MatPow": (lambda: self._bat ** self._bat),
            "Max": (lambda: np.max(self._one)),
            "Min": (lambda: np.min(self._one)),
            "Mod": (lambda: self._one % self._bat),
            "Mul": (lambda: self._bat * self._bat),
            "Nand": (lambda: not np.all(self._true)),
            "Nor": (lambda: not np.any(self._false)),
            "Not": (lambda: ~ self._true),
            "Or": (lambda: self._false | self._false),
            "Piecewise": (lambda: ... if True else ...),
            "Pow": (lambda: self._two ** self._bat),
            "StrictGreaterThan": (lambda: self._one > self._one),
            "StrictLessThan": (lambda: self._one < self._one),
            "Transpose": (lambda: np.transpose(self._zero)),
            "Xor": (lambda: (self._false|self._true) & ~(self._false&self._true)),
            "acos": (lambda: np.arccos(self._bat)),
            "acosh": (lambda: np.arccosh(self._two)),
            "arg": (lambda: np.angle(self._comp)),
            "asin": (lambda: np.arcsin(self._bat)),
            "asinh": (lambda: np.arcsinh(self._bat)),
            "atan": (lambda: np.arctan(self._bat)),
            "atan2": (lambda: np.arctan2(self._bat, self._one)),
            "atanh": (lambda: np.arctanh(self._bat)),
            "conjugate": (lambda: np.conjugate(self._comp)),
            "cos": (lambda: np.cos(self._bat)),
            "cosh": (lambda: np.cosh(self._bat)),
            "exp": (lambda: np.exp(self._bat)),
            "im": (lambda: np.imag(self._comp)),
            "log": (lambda: np.log(self._bat)),
            "re": (lambda: np.real(self._comp)),
            "sign": (lambda: np.sign(self._bat)),
            "sin": (lambda: np.sin(self._bat)),
            "sinc": (lambda: np.sinc(self._bat)),
            "sinh": (lambda: np.sinh(self._bat)),
            "sqrt": (lambda: np.sqrt(self._bat)),
            "tan": (lambda: np.tan(self._bat)),
            "tanh": (lambda: np.tanh(self._bat)),
            "eval": (lambda: np.float64("0.123456789e+01")),
            "aloc": (lambda: np.zeros(shape=(1000, 1000))),
            "div": (lambda: 1/self._bat),
            "**2": (lambda: self._bat**2),
        }

        self.load_save()

    def load_save(self):
        """
        Charge le fichier qui contient les resultats.
        """
        dirname = os.path.join(os.path.dirname(os.path.abspath(laue.__file__)), "data")
        file = os.path.join("timecost.pickle")
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.__setstate__(pickle.load(f))
        else:
            with open(file, "wb") as f:
                pickle.dump(self.__getstate__(), f)

    def atom_cost(self, key):
        """
        Retourne le cout de l'opperateur.
        """
        if key in self.costs:
            return self.costs[key]
        if hasattr(self, "_tests"):
            if key in self._tests:
                self.costs[key] = timeit.timeit(self._tests[key], number=100)
                return self.costs[key]
        raise ValueError(f"L'opperation {key} est inconnue.")

    def branch_cost(self, branch):
        """
        Le cout brut de l'expression sympy sans cse.
        """
        if isinstance(branch, (sympy.Atom, numbers.Number)):
            return self.atom_cost("eval")

        if isinstance(branch, (sympy.Add, sympy.And, sympy.MatAdd, sympy.MatMul,
                sympy.MatPow, sympy.Mul, sympy.Nand, sympy.Nor, sympy.Or, sympy.Xor)):
            op_cost = self.atom_cost(type(branch).__name__) * (len(branch.args)-1)
            if multiprocessing.current_process().name == "MainProcess":
                with Pool() as pool:
                    return sum(pool.map(self.branch_cost, branch.args)) + op_cost
            else:
                return sum((self.branch_cost(e) for e in branch.args)) + op_cost

        if isinstance(branch, sympy.Pow):
            if isinstance(branch.exp, sympy.Number):
                if round(branch.exp, 5) == 2:
                    return self.branch_cost(branch.base) + self.atom_cost("**2")
                if round(branch.exp, 5) == .5:
                    return self.branch_cost(branch.base) + self.atom_cost("sqrt")
                if round(branch.exp, 5) == -.5:
                    return self.branch_cost(branch.base) + self.atom_cost("div") + self.atom_cost("sqrt")
                if round(branch.exp, 5) == -1:
                    return self.branch_cost(branch.base) + self.atom_cost("div")
                if round(branch.exp, 5) == -2:
                    return self.branch_cost(branch.base) + self.atom_cost("div") + self.atom_cost("**2")
            return self.branch_cost(branch.base) + self.branch_cost(branch.exp) + self.atom_cost("Pow")

        if isinstance(branch, sympy.Piecewise):
            op_cost = self.atom_cost("Piecewise") * (len(branch.args)-1)
            return sum(
                self.branch_cost(val) + sum(self.branch_cost(a) for a in cond.args)
                for val, cond in branch.args
                )/len(branch.args) + op_cost

        if multiprocessing.current_process().name == "MainProcess":
            with Pool() as pool:
                return sum(pool.map(self.branch_cost, branch.args)) + self.atom_cost(type(branch).__name__)
        else:
            return sum((self.branch_cost(e) for e in branch.args)) + self.atom_cost(type(branch).__name__)

    def __call__(self, expr):
        """
        Le cout de l'expression avec cse.
        """
        defs, rvs = sympy.cse(expr)
        if multiprocessing.current_process().name == "MainProcess":
            with Pool() as pool:
                cost = sum(pool.map(
                    self.branch_cost,
                    itertools.chain(
                        (e for var, e in defs),
                        (e for e in rvs))
                    )) + len(defs)*self.atom_cost("aloc")
        else:
            cost = (
                sum(self.branch_cost(e) for var, e in defs)
              + sum(self.branch_cost(e) for e in rvs)
              + len(defs)*self.atom_cost("aloc"))
        return cost

    def __getstate__(self):
        if hasattr(self, "_tests"):
            for key in self._tests:
                self.atom_cost(key)
        return self.costs

    def __setstate__(self, state):
        self.costs = state

class Lambdify:
    """
    ** Permet de manipuler plus simplement une fonction. **
    """
    def __init__(self, args, expr, *, verbose=False, _simp_expr=None):
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
        self.verbose = verbose

        # Preparation vectoriele.
        self._simp_expr = _simp_expr
        if self._simp_expr is None:
            self._simp_expr = simplify(self.expr, measure=TimeCost(), verbose=verbose)
        self.fct = lambdify(self.args, self._simp_expr, cse=True, modules="numpy")
        try:
            self.fct_numexpr = lambdify(self.args, evalf(self._simp_expr, n=15), cse=True, modules="numexpr")
        except (ImportError, TypeError, RuntimeError):
            self.fct_numexpr = None

    def __str__(self, *, name="lambdifygenerated", bloc="numpy"):
        '''
        ** Offre une representation explicite de la fonction. **

        Parameters
        ----------
        name : str
            Le nom a donner a la fonction.
        bloc : str
            La partie du code a impromer. Permet de selectionner la fonction.

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos, pi
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> print(Lambdify([x, y], pi*cos(x + y) + x + y), end="")
        def _lambdifygenerated_numpy(x, y):
            """Perform calculations in small float using the numpy module."""
            x0 = x + y
            _0 = x0 + 3.14159265358979*cos(x0)
            return _0
        >>>
        '''
        assert isinstance(name, str), f"'name' has to be str, not {type(name).__name__}."
        assert bloc in {"main", "numpy", "numpy128", "numexpr", "sympy"}
        import re

        # Code numexpr.
        if bloc == "numexpr":
            if self.fct_numexpr is not None:
                code = self.fct_numexpr.__doc__.split("\n")
                code[0] = code[0].replace("_lambdifygenerated", f"_{name}_numexpr")
                code.insert(1, '    """Perform calculations in float64 using the numexpr module."""')
            else:
                code = []

        # Code < float 64
        elif bloc == "numpy":
            code = lambdify(self.args, evalf(self._simp_expr, n=15),
                cse=True, modules="numpy").__doc__.split("\n")
            code[0] = code[0].replace("_lambdifygenerated", f"_{name}_numpy")
            code.insert(1, '    """Perform calculations in small float using the numpy module."""')

        # Code float 128
        elif bloc == "numpy128":
            f_mod = re.compile(r"""(?:[+-]*
                (?:
                  \. [0-9]+ (?:_[0-9]+)*
                  (?: e [+-]? [0-9]+ (?:_[0-9]+)* )?
                | [0-9]+ (?:_[0-9]+)* \. (?: [0-9]+ (?:_[0-9]+)* )?
                  (?: e [+-]? [0-9]+ (?:_[0-9]+)* )?
                | [0-9]+ (?:_[0-9]+)*
                  e [+-]? [0-9]+ (?:_[0-9]+)*
                ))""", re.VERBOSE | re.IGNORECASE) # Model d'un flottant.
            code_str = self.fct.__doc__
            code_str = re.sub(f_mod,
                lambda m: (f"float128({repr(m.group())})" if len(m.group()) >= 15 else m.group()),
                code_str)
            code = code_str.split("\n")
            code[0] = code[0].replace("_lambdifygenerated", f"_{name}_numpy128")
            code.insert(1, '    """Perform calculations in float128 using the numpy module."""')

        # Expression formelle
        elif bloc == "sympy":
            code = []
            code.append(f"def _{name}_sympy():")
            code.append( '    """Returns the tree of the sympy expression."""')
            code.append(f"    {', '.join(self.args_name)} = symbols('{' '.join(self.args_name)}')")
            code.append(f"    return {self.expr}")
            code.append( "")

        # Fonction principale Equivalent as self.__call__.
        elif bloc == "main":
            code = []
            code.append(f"def {name}(*args, **kwargs):")
            code.append( '    """')
            code.append( "    ** Choose the most suitable function according to")
            code.append( "    the type and size of the input data. **")
            code.append( "")
            code.append( "    Parameters")
            code.append( "    ----------")
            code.append( "    *args")
            code.append( "        Les parametres ordonnes de la fonction.")
            code.append( "    **kwargs")
            code.append( "        Les parametres nomes de la fonction. Ils")
            code.append( "        ont le dessus sur les args en cas d'ambiguite.")
            code.append( '    """')

            code.append( "    assert len(args) <= %d, f'The function cannot take {len(args)} arguments.'"
                                                % len(self.args))
            code.append( "    assert not set(kwargs) - {%s}, f'You cannot provide {kwargs}.'"
                                                      % ", ".join(repr(a) for a in self.args_name))
            code.append( "    if not args and not kwargs:")
            code.append(f"        from laue.data.sympy_lambdify import _{name}_sympy")
            code.append(f"        return _{name}_sympy()")

            code.append( "    args = list(args)")
            code.append(f"    if len(args) < {len(self.args)}:")
            code.append(f"        args += sympy.symbols(' '.join({self.args_name}[len(args):]))")
            code.append( "    if kwargs:")
            code.append( "        for arg, value in kwargs.items():")
            code.append(f"            args[{self.args_position}[arg]] = value")

            code.append( "    if any(isinstance(a, sympy.Basic) for a in args):")
            code.append( "        sub = {arg: value for arg, value in zip(%s, args)}" % self.args_name)
            code.append( "        from laue.utilities.lambdify import subs")
            code.append(f"        from laue.data.sympy_lambdify import _{name}_sympy")
            code.append(f"        return subs(_{name}_sympy(), sub)")

            if hasattr(np, "float128"):
                code.append( "    if any(a.dtype == np.float128 for a in args if isinstance(a, np.ndarray)):")
                code.append(f"        from laue.data.numpy128_lambdify import _{name}_numpy128")
                code.append(f"        return _{name}_numpy128(*args)")
            if self.fct_numexpr is not None:
                code.append( "    if (")
                code.append( "            (max((a.size for a in args if isinstance(a, np.ndarray)), default=0) >= 157741)")
                code.append( "            and all(a.dtype == np.float64 for a in args if isinstance(a, np.ndarray))")
                code.append( "        ):")
                code.append(f"        from laue.data.numexpr_lambdify import _{name}_numexpr")
                code.append(f"        return _{name}_numexpr(*args)")
            code.append(f"    from laue.data.numpy_lambdify import _{name}_numpy")
            code.append(f"    return _{name}_numpy(*args)")
            code.append( "")

        else:
            raise KeyError

        return "\n".join(code)

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

    def __getstate__(self):
        """
        ** Extrait l'information serialisable. **

        Examples
        --------
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> l = Lambdify([x, y], cos(x + y) + x + y)
        >>> l.__getstate__()
        ([x, y], x + y + cos(x + y))
        >>>
        """
        if self.expr == self._simp_expr:
            return (self.args, self.expr, self.verbose)
        return (self.args, self.expr, self.verbose, self._simp_expr)

    def __setstate__(self, state):
        """
        ** Instancie l'objet a partir de l'etat. **

        Examples
        --------
        >>> import pickle
        >>> from sympy.abc import x, y; from sympy import cos
        >>> from laue.utilities.lambdify import Lambdify
        >>>
        >>> l = Lambdify([x, y], cos(x + y) + x + y)
        >>> l
        Lambdify([x, y], x + y + cos(x + y))
        >>> pickle.loads(pickle.dumps(l))
        Lambdify([x, y], x + y + cos(x + y))
        >>>
        """
        self.__init__(state[0], state[1], verbose=state[2], _simp_expr=state[-1])
