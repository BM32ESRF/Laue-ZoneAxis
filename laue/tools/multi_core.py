#!/usr/bin/env python3

"""
** Outils pour le paralelisme. **
---------------------------------

* Aide a la serialisation des fonctions (pickle).
* Permet de gerer plusieur instances asynchrones de generateurs en multi-threads.
"""

import hashlib
import math
import multiprocessing
import os
import time

import cloudpickle


def limited_imap(pool, func, iterable, **kwargs):
    """
    ** Same as ``Pool.imap`` with limited buffer. **

    La fonction ``Pool.imap`` du module multiprocessing epuise
    tant qu'elle peut l'iterable d'entree, et accumule les resultat
    dans une memoir tampon. Seulement, elle ne se preocupe
    pas de la memoire disponible ni des autres processus.
    Ici, les calcul sont fait en economisant les ressources
    disponible de facon a accroitre les peformances.

    Parameters
    ----------
    pool : multiprocessing.pool.Pool
        Pool de ``multiprocessing.Pool()``.
    func : callable
        La fonction serialisable avec pickle qui sera evaluee.
    iterable : iterable
        Cede sucessivement les argument a fournir a ``func``.
    **kwargs
        See ``multiprocessing.Pool().imap``.

    Yields
    ------
    result
        Cede peu a peu les resultats de la fonction ``func``.
    """
    class Regulator:
        """
        Permet de reguler le flot d'un bloc.
        """
        def __init__(self, pool, iterable):
            self.pool = pool
            self.iterable = iterable
            self.nbr_yields = 0 # Le nombre de resultats cedes.
            self.nbr_args = 0 # Le nombre d'arguments pompes.
            self.max_tasks = 2*os.cpu_count() # Nombre de taches maximales en cours de calcul.

        def __iter__(self):
            """
            Cede les arguments au compte gouttes.
            """
            for args in self.iterable:
                while True: # Permet d'attendre en cas de besoin.
                    buff_size = self.nbr_args - self.nbr_yields
                    if buff_size < self.max_tasks:
                        break
                    if buff_size > 10*self.max_tasks: # Si il y a suffisement de resultats en avance.
                        time.sleep(.1) # On fait une grande pause.
                        continue # Et on attend que ca se decante.
                    cpu = min(psutil.cpu_percent(interval=0.05, percpu=True))
                    mem = psutil.virtual_memory().percent
                    if cpu < 50 and mem < 75: # Si il y a suffisement de ressources.
                        break

                self.nbr_args += 1
                yield args

        def imap(self, func, **kwargs):
            """
            Cede les resultats.
            """
            for res in self.pool.imap(func, self, **kwargs):
                self.nbr_yields += 1
                yield res
    
    try:
        import psutil
    except ImportError:
        import logging
        logging.warn("'psutil' n'est installer, il est impossible de "
            "gerer poprement les ressources.")
        psutil = None
    
    if psutil is None:
        yield from pool.imap(func, iterable, **kwargs)
    else:
        regulator = Regulator(pool, iterable)
        yield from regulator.imap(func, **kwargs)

def pickleable_method(args, serialize=False):
    """
    ** Permet de serialiser une methode. **

    Notes
    -----
    Comme l'utilisateur ne doit pas utiliser cette fonction, il n'y
    a pas de verifications sur les entree de facon a privilegier la performance.

    Parameters
    ----------
    args : tuple
        * args[0] => func, La fonction a executer. Si c'est une methode, il est
        possible de fournir ``ClasseName.methode`` au lieu de ``self.methode``.
        * args[1] => self, Serialiser ou non, c'est le premiers argument.
        * args[2] (facultativ) => kwargs, Le dictionaire des parametres nomes.
    serialize : bool
        Serialize le resultat (si True), sinon laisse implicitement
        pickle le faire (False). Si pickle est capable de le faire, il
        faut lui laisser gerer ca car c'est plus efficace.
    """
    if len(args) == 2:
        (func, self), kwargs = args, {}
    else:
        func, self, kwargs = args
    if isinstance(self, bytes):
        self = cloudpickle.loads(self)
    return (
                (lambda x: cloudpickle.dumps(x))
                if serialize else
                (lambda x: x)
            )(func(self, **kwargs))

def prevent_generator_size(min_size=1, max_size=math.inf):
    """
    ** Controle que le generateur contient le bon nombre d'elements. **

    C'est un decorateur qui decore une fonction (generateur).

    Parameters
    ----------
    min_size : int
        Le nombre minimum d'elements que doit ceder le generateur avant qu'il ne soit epuise.
    max_size : int
        Le nombre maximum d'elements cedes avant de lever l'exception.

    Raises
    ------
    GeneratorExit
        Si les conditions ne sont pas respectees.
    """
    assert isinstance(min_size, int), f"'min_size' has to be int, not {type(min_size).__name__}."
    assert isinstance(max_size, int) or max_size == math.inf, \
        f"'max_size' has to be int, not {type(max_size).__name__}."
    assert max_size >= min_size, ("'max_size' doit etre plus grand ou egal a 'min_size'. "
        f"Or ils valent respectivement {max_size} et {min_size}.")
    assert min_size >= 0, f"'min_size' ne doit pas etre negatif ({min_size})."

    def decorator(func):
        def generator(*args, **kwargs):
            i = -1
            for i, element in enumerate(func(*args, **kwargs)):
                yield element
                if i >= max_size:
                    raise GeneratorExit(f"Le generateur {func} ne doit pas ceder plus de "
                        f"{max_size} elements. Or il tente d'en ceder un {i+1}eme.")
            if i+1 < min_size:
                raise GeneratorExit(f"Le generateur {func} doit ceder au moins {min_size} "
                    f"elements. Or il n'en a cede que {i+1}.")

        return generator
    return decorator

def reduce_object(obj, attrs=None):
    """
    ** Reconstitue un objet partiel. **

    Le but est de rendre un objet plus facilement serialisable et
    aussi de l'alleger pour perdre moins de temps avec la
    serialisation puis la deserialisation.

    Parameters
    ----------
    obj : objet
        Instance de la classe que l'on shouaite reduire.
    attrs : iterable
        Les noms des attributs qui resteront presents dans l'objet final.
        Si il ne sont pas precise, seul les attributs serialisables sont gardes.

    Examples
    --------
    >>> from laue.tools.multi_core import reduce_object
    >>>
    >>> class Foo:
    ...     def __init__(self):
    ...         self.attr1 = 1
    ...         self.attr2 = 2
    ...     def meth(self):
    ...         pass
    ...
    >>> obj = Foo()
    >>>
    >>> little_obj = reduce_object(obj, ["attr1"])
    >>> hasattr(little_obj, "attr1")
    True
    >>> hasattr(little_obj, "attr2")
    False
    >>> hasattr(little_obj, "meth")
    True
    >>> little_obj.attr1
    1
    >>> type(little_obj.meth)
    <class 'method'>
    >>>
    """
    class Partial(type(obj)):
        def __init__(self):
            pass

    if attrs is None:
        all_attrs = [attr for attr in dir(obj)
            if type(getattr(obj, attr)).__name__ != "method"
            and not attr.startswith("__")]
        attrs = []
        for attr in all_attrs:
            try:
                cloudpickle.dumps(getattr(obj, attr))
            except (TypeError, RuntimeError):
                continue
            else:
                attrs.append(attr)
        print(attrs)

    partial_obj = Partial()
    for attr in attrs:
        setattr(partial_obj, attr, getattr(obj, attr))
    return partial_obj

    # class Constructor:
    #     """
    #     Methaclasse qui fabrique de nouvelle instances sur mesure.
    #     """
    #     def __new__(cls, obj, attrs):
    #         name = f"Partial{type(obj).__name__}"
            
    #         # Separation des attributs et des methodes.
    #         meth_name = {attr for attr in attrs if type(getattr(obj, attr)).__name__ == "method"}
    #         attrs = set(attrs) - meth_name
            
    #         # Creation d'un objet ayant que les methodes.
    #         empty_obj = NoAttrs()
    #         # for attr in attrs:
    #         #     setattr(emty_obj, attr, getattr(obj, attr))
    #         methodes = {name: getattr(empty_obj, name) for name in meth_name}

    #         # Ajout des attributs.
    #         partial_obj = type(name, (), methodes)
    #         for attr in attrs:
    #             setattr(partial_obj, attr, getattr(obj, attr))
    #         return partial_obj

    # return Constructor(obj, attrs)

class RecallingIterator:
    """
    ** Permet d'iterer plusieurs instances d'un unique generateur. **

    Examples
    --------
    Example avec une classe.
    >>> from laue.tools.multi_core import RecallingIterator
    >>>
    >>> class Foo:
    ...     def __init__(self):
    ...         self.it = iter(range(5))
    ...     def __iter__(self):
    ...         yield from RecallingIterator(self.it, mother=self)
    ...
    >>> foo = Foo()
    >>> for a, b in zip(foo, foo):
    ...     a, b
    ... 
    (0, 0)
    (1, 1)
    (2, 2)
    (3, 3)
    (4, 4)
    >>> list(foo)
    [0, 1, 2, 3, 4]
    >>>

    Example sans classe
    >>> it = iter(range(5))
    >>> for a, b in zip(RecallingIterator(it), RecallingIterator(it)):
    ...     a, b
    ... 
    (0, 0)
    (1, 1)
    (2, 2)
    (3, 3)
    (4, 4)
    >>> list(RecallingIterator(it))
    [0, 1, 2, 3, 4]
    >>>
    """
    def __init__(self, base_iterator, *, mother=None):
        """
        Paremeters
        ----------
        mother : object
            Instance de la classe dans laquelle on met cet iterateur.
            Si il est omis, la memoire est globale et ne sera pas netoyee
            par le ramasse-miette. Il est conseille si possible fournir 'mother'.
        base_iterator : iterator
            L'iterateur epuisable, qui ne doit en tout et pour tout
            etre parcouru qu'une seule fois.
        """
        self.mother = mother
        self.base_iterator = base_iterator
        self.signature = hashlib.md5(id(base_iterator).to_bytes(16, "big")).hexdigest()

        self.stape = 0 # Le rang de l'element suivant a ceder.

        # Mise en place du verrou.
        lock_name = f"_lock_{self.signature}"
        if self.mother is not None:
            if not hasattr(self.mother, lock_name):
                setattr(self.mother, lock_name, multiprocessing.Lock())
            self.lock = getattr(self.mother, lock_name)
        else:
            if lock_name not in globals():
                globals()[lock_name] = multiprocessing.Lock()
            self.lock = globals()[lock_name]

        # Mise en place de la memoire pour reiterer.
        buffer_name = f"_buffer_{self.signature}"
        if self.mother is not None:
            if not hasattr(self.mother, buffer_name):
                setattr(self.mother, buffer_name, [])
            self.buffer = getattr(self.mother, buffer_name)
        else:
            if buffer_name not in globals():
                globals()[buffer_name] = []
            self.buffer = globals()[buffer_name]

    def __iter__(self):
        """
        ** Permet d'etre mis dans une boucle for. **
        """
        return self

    def __next__(self):
        """
        ** Itere de facon intrementale. **

        Raises
        ------
        StopIteration
            Quand tous les paquets sont cedes.
        """
        with self.lock:
            # Si il ne faut pas iterer 'base_iterator'
            if self.stape < len(self.buffer):
                self.stape += 1
                return self.buffer[self.stape-1]

            # Si il faut iterer.
            while True:
                try:
                    element = next(self.base_iterator)
                except ValueError: # precisement: generator already executing
                    time.sleep(.1) # On attend 100 ms avant de retenter.
                else:
                    self.buffer.append(element)
                    self.stape += 1
                    return element
