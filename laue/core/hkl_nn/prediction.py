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

from fast_histogram import histogram1d
import h5py
import numpy as np


__pdoc__ = {"Predictor.__call__": True}


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
        (material_1,), hkl_data_path = select(r"^hkl_data_(?P<material>\w+)\.pickle$")
        (material_2,), weights_path = select(r"^my_model_(?P<material>\w+)\.h5$")
        _, angbin_path = select(r"^MOD_grain_hkl_data_angbin.npz$")

        if material_1 != material_2:
            raise ValueError("Il y a ambiguite. "
                f"Le materiaux c'est {material_1} ou bien {material_2}?")
        self.material = material_1

        # Load data.
        with open(hkl_data_path, "rb") as hkl_data_file:
            self.hkl_all_class = pickle.load(hkl_data_file)[5]
        self.wb = self._read_hdf5(weights_path)
        self.temp_key = list(self.wb.keys())
        self.hkl_data = np.load(angbin_path)["arr_0"]
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

    def _predict(self, x):
        """
        ** Help for ``Predictor.__call__``. **
        """
        softmax = lambda x: (np.exp(x).T / np.sum(np.exp(x).T, axis=0)).T
        # first layer
        l0 = np.dot(x, self.wb[self.temp_key[1]]) + self.wb[self.temp_key[0]] 
        l0 = np.maximum(0, l0) ## ReLU activation
        # Second layer
        l1 = np.dot(l0, self.wb[self.temp_key[3]]) + self.wb[self.temp_key[2]] 
        l1 = np.maximum(0, l1) ## ReLU activation
        # Third layer
        l2 = np.dot(l1, self.wb[self.temp_key[5]]) + self.wb[self.temp_key[4]]
        l2 = np.maximum(0, l2) ## ReLU activation
        # Output layer
        l3 = np.dot(l2, self.wb[self.temp_key[7]]) + self.wb[self.temp_key[6]]
        l3 = softmax(l3) ## output softmax activation
        return l3

    def __call__(self, theta_chi):
        """
        ** Estime les hkl. **

        Parameters
        ----------
        theta_chi : np.ndarray
            La matrice des thetas et des chi.
            shape = (2, n), avec n le nombre de spots.

        Returns
        -------
        hkl
            Le vecteur des indices hkl
        score
            Le vecteur des fiabilitees.
        """
        sorted_data = theta_chi.transpose()
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))
        
        codebars_all = []        
        spots_in_center = np.arange(0,len(data_theta))

        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            # codebars = np.histogram(spotangles, bins=angbins)[0]
            codebars = histogram1d(spotangles, range=[min(self.angbins), max(self.angbins)], bins=len(self.angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars / max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## Do prediction of all spots at once
        prediction = self._predict(codebars)
        max_pred = np.max(prediction, axis=1)
        class_predicted = np.argmax(prediction, axis=1)
        predicted_hkl = self.hkl_data[class_predicted]
        ## return predicted HKL and their softmax confidence
        return predicted_hkl, max_pred
