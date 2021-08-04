#!/usr/bin/env python3

"""
** Deep learning for HKL classification. **
-------------------------------------------

Now can predict with > 95% accuracy for multi grain Laue Patterns.
If you have model save files; go to cell 45 to load and start prediction
Pros: Impressive speed for prediction; results not dependent on
the statistical descriptor (ex: correlation function or distance measurment).
Cons: Building reliable test data can take few hours
(this will significantly increase for less symmetry crystals).
"""

import os
import pickle
import re

import h5py
import numpy as np


class Predictor:
    """
    Cette classe permet de charger le model
    une bonne fois pour toutes.
    """
    def __init__(self, model_directory):
        """
        Parameters
        ----------
        model_directory : str
            Le chemin du repertoire qui contient les donnees du model.
        """
        def select(reg_expr):
            """Cherche le bon fichier et s'assure qu'il soit unique."""
            files = os.listdir(self.model_directory)
            found = [
                (pred.groups(), os.path.join(self.model_directory, files[i]))
                for i, pred in enumerate(
                    re.search(reg_expr, file)
                    for file in os.listdir(self.model_directory))
                if pred is not None]
            if not found:
                raise FileNotFoundError("Le dossier doit contenir un fichier respectant "
                    f"l'expression reguliere: {reg_expr}")
            if len(found) > 1:
                raise ValueError(f"Seul 1 fichier doit matcher avec {reg_expr}. "
                    f"Or {', '.join(f for _, f in found)} sont candidats.")
            return found.pop()
        
        # Verifications.
        assert isinstance(model_directory, str), ("'model_directory' "
            f"has to be str, not {type(model_directory).__name__}.")
        assert os.path.isdir(model_directory), \
            f"{repr(model_directory)} doit etre un dossier existant."

        self.model_directory = model_directory
        
        # File search.
        (material_1,), classhkl_path = select(r"^classhkl_data_(?P<material>\w+)\.pickle$")
        (material_2,), weights_path = select(r"^my_model_(?P<material>\w+)\.h5$")
        _, angbin_path = select(r"^MOD_grain_classhkl_angbin.npz$")

        if material_1 != material_2:
            raise ValueError("Il y a ambiguite. "
                f"Le materiaux c'est {material_1} ou bien {material_2}?")
        self.material = material_1

        # Load data.
        with open(classhkl_path, "rb") as classhkl_file:
            self.hkl_all_class = pickle.load(classhkl_file)[5]
        self.wb = self._read_hdf5(weights_path)
        self.temp_key = list(self.wb.keys())
        self.classhkl = np.load(angbin_path)["arr_0"]
        self.angbins = np.load(angbin_path)["arr_1"]

    def _read_hdf5(self, path):
        """Read a specific hdf5 file."""
        weights = {}
        keys = []
        with h5py.File(path, 'r') as f: # open file
            f.visit(keys.append) # append all keys to list
            for key in keys:
                if ':' in key: # contains data if ':' in key
                    weights[f[key].name] = f[key].value
        return weights
