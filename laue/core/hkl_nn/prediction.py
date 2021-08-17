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

        Examples
        --------
        >>> from laue.core.hkl_nn.prediction import Predictor
        >>> pred = Predictor("laue/data/Zr02")
        >>>
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
        _, angbin_path = select(r"^MOD_grain_classhkl_angbin.npz$")

        if material_1 != material_2:
            raise ValueError("Il y a ambiguite. "
                f"Le materiaux c'est {material_1} ou bien {material_2}?")
        self.material = material_1

        # Load data.
        with open(hkl_data_path, "rb") as hkl_data_file:
            self.hkl_all_class = pickle.load(hkl_data_file)[0] # Pourquoi le metre dans une liste de 1 element?
        self.wb = self._read_hdf5(weights_path)
        self.temp_key = list(self.wb.keys()) # L'ordre n'est pas garanti!
        self.hkl_data = np.load(angbin_path)["arr_0"]
        self.angbins = np.load(angbin_path)["arr_1"]

    def _read_hdf5(self, path):
        """Read a specific hdf5 file."""
        weights = {} # Si on se sert de l'index des clef, autant utiliser directement une liste?
        keys = []
        with h5py.File(path, "r") as f: # open file
            f.visit(keys.append) # append all keys to list
            for key in keys:
                if ":" in key: # contains data if ':' in key
                    weights[f[key].name] = f[key].__array__() # Je ne comprend pas bien cette mise en forme?
                    # weights[f[key].name] = f[key].value # AttributeError: 'Dataset' object has no attribute 'value'
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

        Examples
        --------
        >>> import numpy as np
        >>> from laue.core.hkl_nn.prediction import Predictor
        >>> pred = Predictor("laue/data/Zr02")
        >>> theta_chi = np.array(
        ...   [[ 2.52627945e+01,  2.60926781e+01,  2.65539875e+01,
        ...      2.81886940e+01,  2.88142796e+01,  2.92087307e+01,
        ...      3.04267445e+01,  3.20891838e+01,  3.23967743e+01,
        ...      3.43747025e+01,  3.50725060e+01,  3.51420059e+01,
        ...      3.53640709e+01,  3.66132927e+01,  3.71793442e+01,
        ...      3.78125343e+01,  3.82845306e+01,  3.89372597e+01,
        ...      3.97087402e+01,  4.00737114e+01,  4.07332764e+01,
        ...      4.12437706e+01,  4.04045906e+01,  4.12707405e+01,
        ...      4.20407639e+01,  4.40923157e+01,  4.43046799e+01,
        ...      4.46586914e+01,  4.49168205e+01,  4.54237938e+01,
        ...      4.57435112e+01,  4.64431877e+01,  4.68662910e+01,
        ...      4.70701141e+01,  4.75698853e+01,  4.77383308e+01,
        ...      4.84347076e+01,  4.83345680e+01,  4.82600441e+01,
        ...      4.83319550e+01,  4.89955254e+01,  4.98372574e+01,
        ...      4.80468826e+01,  4.91234589e+01,  5.12316246e+01,
        ...      5.14579887e+01,  5.48247643e+01,  5.42389297e+01,
        ...      5.50067596e+01,  5.47498512e+01,  5.53464012e+01,
        ...      5.44665947e+01,  5.51647453e+01,  5.69880180e+01,
        ...      5.69231071e+01,  5.71547623e+01,  5.59031296e+01,
        ...      5.65157928e+01,  5.54328194e+01,  5.79304810e+01,
        ...      5.83737488e+01,  5.91063004e+01,  5.93900108e+01,
        ...      5.98970032e+01,  5.77238159e+01,  5.97812843e+01,
        ...      5.84865570e+01,  6.03050117e+01,  6.24293098e+01,
        ...      6.18709679e+01,  6.22257271e+01,  6.39516373e+01,
        ...      6.19101753e+01,  6.25465889e+01,  6.08266754e+01,
        ...      6.68958588e+01,  6.72588577e+01,  6.76704407e+01],
        ...    [-2.53227158e+01, -2.17357197e+01,  2.34524479e+01,
        ...      7.92672694e-01, -1.83965931e+01,  2.00298729e+01,
        ...      7.78525651e-01, -1.39836121e+01,  1.55002489e+01,
        ...     -3.47049446e+01,  3.63148613e+01, -9.30009365e+00,
        ...      1.07037458e+01, -2.74188271e+01,  2.88973942e+01,
        ...     -2.24778461e+01,  2.38802395e+01, -3.84828644e+01,
        ...      4.00165977e+01, -3.23755569e+01,  3.37935753e+01,
        ...     -4.06398621e+01,  6.06966138e-01,  2.91247730e+01,
        ...      4.21363029e+01, -9.69736004e+00,  1.07738695e+01,
        ...     -1.17179070e+01,  1.27956800e+01, -1.47250223e+01,
        ...      1.57742653e+01, -1.95509968e+01,  2.06181507e+01,
        ...     -2.32521973e+01,  2.43160992e+01, -2.84838123e+01,
        ...      4.39006448e-01,  2.95768604e+01, -3.62519379e+01,
        ...     -3.96765099e+01,  3.74329453e+01,  3.96892816e-01,
        ...     -4.83194771e+01,  4.09185829e+01, -1.01135015e+01,
        ...      1.08719740e+01,  2.61049420e-01, -2.40735455e+01,
        ...     -1.55187674e+01,  2.47712955e+01,  1.60778942e+01,
        ...     -3.42320824e+01,  3.51054649e+01,  1.88722670e-01,
        ...     -1.05438442e+01,  1.09418879e+01, -2.96021423e+01,
        ...      3.03011723e+01, -4.09050407e+01, -2.08004246e+01,
        ...      2.12467175e+01, -1.28589134e+01,  1.31161022e+01,
        ...      6.30038828e-02, -3.78738708e+01, -2.49110565e+01,
        ...      3.86065025e+01,  2.52701626e+01, -4.97089103e-02,
        ...     -1.63510494e+01,  1.63965435e+01, -1.22255214e-01,
        ...     -3.07791405e+01,  3.10674877e+01, -4.21732368e+01,
        ...     -1.72333164e+01,  1.67443314e+01, -1.41095228e+01]])
        >>> hkl, scores = pred(theta_chi)
        >>> hkl[:4]
        array([[ -7.,  -3.,   0.],
               [ -5.,   3.,  -5.],
               [ -5.,   3.,  -5.],
               [ -5.,   8.,   0.]])
        >>> np.round(scores[:4], 2)
        ...
        >>>
        """
        sorted_data = theta_chi.transpose()

        # from laue.spot import distance
        # tabledistancerandom = distance(sorted_data, sorted_data, space="cosine")
        import LaueTools.generaltools as GT
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

        codebars_all = []        
        spots_in_center = np.arange(0, len(sorted_data))

        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i) # removing the self distance
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
