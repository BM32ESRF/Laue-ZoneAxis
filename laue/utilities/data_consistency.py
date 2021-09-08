#!/usr/bin/env python3

"""
** Permet de gerer l'enregistrement des experiences en arriere plan. **
-----------------------------------------------------------------------

Quand on instancie une grosse experience, effectuer un traitement sur toutes
les images peut etre tres long. Or pandant le traitement, on a plus
acces a l'instance de l'experience. Il est donc difficile d'enregistrer
les resultats au fur a mesure. C'est pour cela qu'on ajoute a la classe
``laue.experiment.base_experiment.Experiment`` une interface chargee d'enregistrer
l'etat de l'experience pour pouvoir la reprendre a tout moments.
"""

import numbers
import os
import pickle
import threading
import time


__pdoc__ = {"Recordable.__enter__": True}


class Recordable(threading.Thread):
    """
    ** Interface asynchrone gerant la persistance des donnees. **
    """
    def __init__(self, saving_file="experiment_state", compress=False, dt=600, **_kwargs):
        """
        ** Initialise le gestionaire d'enregistrement. **

        Dans le cas ou l'etat de l'ancienne session de travail
        est fournis en parametre, les element de la nouvelle experience
        sont mis a jour.

        Parameters
        ----------
        saving_file : str
            Le path du fichier dans lequel sera enregistre l'etat de l'experience.
        compress : boolean
            Si True, compresse les donnees avec l'algorithme gzip. Cela utilise
            plus de CPU mais reduit un peu la taille des donnees a ecrire.
        dt : number
            Le temps qui separe 2 enregistrements (en secondes).
        """
        assert isinstance(saving_file, str), f"'file' has to be of type str, not {type(saving_file).__name__}."
        assert isinstance(compress, bool), \
            f"'compress' has to be of type bool, not {type(compress).__name__}."
        assert isinstance(dt, numbers.Number), f"'dt' has to be a number, not a {type(dt).__name__}."
        assert dt > 0, f"Le temps vaut {dt}."

        self.saving_file = saving_file
        self.compress = compress
        self.dt = dt

        self._must_die = False

        threading.Thread.__init__(self)

        if os.path.exists(self.saving_file):
            self._load()

    def __enter__(self):
        """
        ** Gestionaire de contexte. **

        Demarre l'activite du thread.

        Examples
        --------
        >>> import os, tempfile
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> saving_file = os.path.join(tempfile.mkdtemp(), "state")
        >>>
        >>> with laue.experiment.base_experiment.Experiment(image, saving_file=saving_file, verbose=2) as experiment:
        ...     pass
        ...
        Starting the thread.
        The thread says: I was just born.
        Sends the thread the order to succumb...
        Recording of the current status...
            OK: the state of the experiment is recorded.
        The thread says: Oh help! I'm dying!
            the thread is killed.
        >>>
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.kill()

    def __del__(self):
        try:
            if self.is_alive():
                self.kill()
        except AssertionError:
            pass

    def kill(self):
        """
        ** Stop the thread activity. **
        """
        if self.verbose:
            print("Sends the thread the order to succumb...")
        self._must_die = True
        while self.is_alive():
            time.sleep(.1)
        if self.verbose:
            print("    the thread is killed.")

    def run(self):
        """
        ** Enregistre indefiniment l'etat de l'experience. **

        Cette methode ne doit pas etre appelee directement car elle
        est bloquante. Pour l'executer en tache de fond il faut invoquer
        ``laue.utilities.data_consistency.Recordable.start``. Ou bien
        l'instanceier via un gestionaire de contexte.
        """
        def pause(dt):
            ti = time.time()
            while not self._must_die and time.time() - ti < dt:
                time.sleep(0.01)

        if self.verbose >= 2:
            print("The thread says: I was just born.")

        while not self._must_die:
            pause(self.dt)
            self.save_state()

        if self.verbose >= 2:
            print("The thread says: Oh help! I'm dying!")

    def start(self):
        """
        ** Demarre l'activite du thread. **

        Elle doit etre appelee au maximum une fois par objet thread.
        Elle fait en sorte que la methode run()
        de l'objet soit invoquee dans un thread de controle separe.

        Cette methode declenchera une RuntimeError si elle
        est appelee plus d'une fois sur le meme objet thread.
        """
        if self.verbose:
            print("Starting the thread.")
        super().start()

    def _load(self):
        """
        ** Met a jour l'etat de l'experience a partir du fichier. **
        """
        if self.verbose:
            print("Updating the current state...")
        with open(self.saving_file, "rb") as f:
            compress = f.read(1)
            if compress == b"\x00":
                state = pickle.load(f)
            elif compress == b"\x01":
                from gzip import decompress
                state = pickle.loads(decompress(f.read()))
            else:
                raise ValueError(r"Le fichier doit commencer par b'\x00' ou b'\x01'."
                    f"Or il commence par {compress}.")
        self.__setstate__(state)
        if self.verbose:
            print("    OK: the attributes are updated")

    def save_state(self):
        """
        ** Enregistre l'etat courant. **
        """
        if self.verbose >= 2:
            print("Recording of the current status...")
        state = self.__getstate__()
        with open(self.saving_file, "wb") as f:
            if self.compress:
                from gzip import compress
                f.write(b"\x01")
                f.write(compress(pickle.dumps(state)))
            else:
                f.write(b"\x00")
                pickle.dump(state, f)
        if self.verbose >= 2:
            print("    OK: the state of the experiment is recorded.")
