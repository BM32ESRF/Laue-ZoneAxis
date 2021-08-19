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
            ## TODO: list or not in list?
            self.hkl_all_class = pickle.load(hkl_data_file)[0] # Pourquoi le metre dans une liste de 1 element?
        self.wb = self._read_hdf5(weights_path)
        self.hkl_data = np.load(angbin_path)["arr_0"]
        self.angbins = np.load(angbin_path)["arr_1"]

        self._hist_range = [self.angbins.min(), self.angbins.max()]

        self.dict_dp={}
        self.dict_dp['kf_direction']='Z>0'
        self.dict_dp['detectorparameters'] = [70.22, 1039.395, 943.57, 0.7478, 0.07186]
        self.dict_dp['detectordistance'] = 70.22
        self.dict_dp['detectordiameter'] = 0.079856*2048
        self.dict_dp['pixelsize'] = 0.079856
        self.dict_dp['dim'] = 2048


    def _read_hdf5(self, path):
        """Read a specific hdf5 file."""
        weights = []
        keys = []
        with h5py.File(path, "r") as f: # open file
            f.visit(keys.append) # append all keys to list
            for key in keys:
                if ":" in key: # contains data if ':' in key
                    try:
                        weights.append(f[key].value)
                    except AttributeError:
                        weights.append(f[key].__array__())
        return weights

    def _predict(self, x):
        """
        ** Help for ``Predictor.__call__``. **
        """
        softmax = lambda x: (np.exp(x).T / np.sum(np.exp(x).T, axis=0)).T
        # first layer
        l0 = np.dot(x, self.wb[1]) + self.wb[0]
        l0 = np.maximum(0, l0) ## ReLU activation
        # Second layer
        l1 = np.dot(l0, self.wb[3]) + self.wb[2]
        l1 = np.maximum(0, l1) ## ReLU activation
        # Third layer
        l2 = np.dot(l1, self.wb[5]) + self.wb[4]
        l2 = np.maximum(0, l2) ## ReLU activation
        # Output layer
        l3 = np.dot(l2, self.wb[7]) + self.wb[6]
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
        ...   [[ 21.068,  21.54 ,  21.721,  22.085,  22.833,  22.498,  22.92 ,
        ...      23.516,  23.8  ,  23.566,  25.803,  24.676,  26.687,  26.613,
        ...      25.173,  27.17 ,  26.285,  26.826,  30.991,  28.051,  27.569,
        ...      27.075,  28.119,  28.615,  28.329,  28.289,  29.284,  28.582,
        ...      28.84 ,  28.993,  29.449,  28.919,  29.532,  29.569,  29.62 ,
        ...      30.36 ,  29.944,  31.596,  31.59 ,  30.285,  30.541,  32.068,
        ...      30.502,  30.576,  31.01 ,  32.364,  30.888,  33.179,  30.968,
        ...      30.885,  31.365,  31.257,  31.146,  31.599,  32.515,  32.316,
        ...      32.271,  32.705,  34.092,  32.716,  34.131,  32.947,  34.338,
        ...      34.427,  34.918,  35.481,  35.568,  35.213,  35.786,  35.867,
        ...      35.805,  36.186,  36.493,  36.853,  36.052,  36.522,  37.187,
        ...      36.768,  36.924,  37.565,  37.341,  37.865,  38.578,  37.872,
        ...      38.716,  38.145,  38.839,  40.927,  40.109,  40.054,  40.948,
        ...      41.086,  41.782,  41.533,  41.332,  41.701,  42.084,  42.202,
        ...      42.748,  42.857,  43.171,  43.412,  43.884,  44.064,  44.122,
        ...      44.647,  44.888,  44.878,  44.929,  45.496,  45.912,  46.018,
        ...      46.855,  46.945,  46.489,  47.46 ,  47.391,  48.26 ,  51.952,
        ...      52.749,  53.691,  55.061,  55.656],
        ...    [ -3.983,  -1.18 ,  -4.09 ,  -1.389,  20.68 ,   8.594,  16.175,
        ...      16.005,  20.645,  14.423, -28.116,  12.645, -30.944, -30.063,
        ...      14.302, -31.013,  22.167,  15.056, -43.934, -25.464, -18.599,
        ...       0.706,  20.823, -25.887,  18.8  ,   7.751,  19.836,  -4.655,
        ...      11.551, -11.033,  17.782,   2.158,  -5.937,  -1.77 ,   1.003,
        ...      18.599,   0.967,  29.994, -26.815,  -6.674,  12.576,  30.234,
        ...      -5.301,  -7.763,  15.665,  30.201,  -7.682, -35.206,   1.735,
        ...      -3.312,  12.502,  -7.999,   1.997,  -3.489,  15.754, -10.219,
        ...      -5.675,  13.657,  29.346,  -5.629,  28.173,   7.485, -15.096,
        ...     -11.839,  16.36 , -21.932, -23.035,  15.824,  16.383,  17.302,
        ...      15.608, -23.142, -21.983, -25.936,  -0.877,   4.743,  24.124,
        ...      -2.843,  -4.011, -22.061,  -8.448,  17.065, -29.782,   2.977,
        ...      25.746,  13.396,  -6.73 , -39.191,  12.943,  -0.268,  15.37 ,
        ...      15.709,  36.166,  20.751,  13.482,  11.316, -17.798,  13.5  ,
        ...       0.247,  11.702,  13.205,  -2.511, -37.506, -15.404, -14.046,
        ...      14.106, -20.632, -33.049, -14.697,  26.14 , -29.167,  -2.601,
        ...     -12.413,  12.397,  49.108,  23.141, -28.3  ,  -1.048,  -4.872,
        ...      -3.471,  10.512, -24.333, -19.858]])
        >>> hkl, scores = pred(theta_chi)
        >>> bests = np.argsort(scores)[-5:]
        >>> bests
        array([75, 84, 87, 76, 54])
        >>> hkl[bests]
        array([[ 7.,  6.,  9.],
               [-3., -2., 10.],
               [ 7.,  3.,  1.],
               [ 8.,  0.,  2.],
               [ 8.,  7.,  6.]])
        >>> scores[bests]
        array([0.99544195, 0.99634276, 0.99950142, 0.99998127, 0.99999844])
        >>>
        """
        from laue.spot import distance

        spots = theta_chi.transpose() # The list of pair (theta, chi).
        codebars = np.array([
            codebar / codebar.max() # normalize the same way as training data
            for codebar in (
                histogram1d(
                    distance(
                        tuple(spot),
                        np.delete(spots, i, axis=0), # removing the self distance
                        space="cosine"),
                    range=self._hist_range,
                    bins=len(self.angbins)-1)
                for i, spot in enumerate(spots))],
            dtype=float)


        # sorted_data = theta_chi.transpose()

        # from laue.spot import distance
        # tabledistancerandom = distance(sorted_data, sorted_data, space="cosine")

        # codebars_all = []
        # spots_in_center = np.arange(0, len(sorted_data))

        # for i in spots_in_center:
        #     spotangles = tabledistancerandom[i]
        #     spotangles = np.delete(spotangles, i) # removing the self distance
        #     # codebars = np.histogram(spotangles, bins=angbins)[0]
        #     codebars = histogram1d(spotangles, range=[min(self.angbins), max(self.angbins)], bins=len(self.angbins)-1)
        #     ## normalize the same way as training data
        #     max_codebars = np.max(codebars)
        #     codebars = codebars / max_codebars
        #     codebars_all.append(codebars)
        # ## reshape for the model to predict all spots at once
        # codebars = np.array(codebars_all)


        ## Do prediction of all spots at once
        prediction = self._predict(codebars)
        max_pred = np.max(prediction, axis=1)
        class_predicted = np.argmax(prediction, axis=1)
        predicted_hkl = self.hkl_data[class_predicted]
        ## return predicted HKL and their softmax confidence
        return predicted_hkl, max_pred

    def generate_orientation(self, s_tth, s_chi, predicted_hkl, spot1_ind, spot2_ind, emax=23):
        """
        Parameters
        ----------
        s_tth : np.ndarray
            The 2*theta vector array.
        s_chi : np.ndarray
            The chi cetor array.
        predicted_hkl : np.array.
            Premiere sortie de self.__call__

        Examples
        --------
        >>> import numpy as np
        >>> from laue.core.hkl_nn.prediction import Predictor
        >>> pred = Predictor("laue/data/Zr02")
        >>>
        >>> params = {'dd': 70.22, 'xcen': 1039.395, 'ycen': 943.57, 'xbet': 0.7478, 'xgam': 0.07186}
        >>> s_tth = 2*np.array(
        ...    [ 21.068,  21.54 ,  21.721,  22.085,  22.833,  22.498,  22.92 ,
        ...      23.516,  23.8  ,  23.566,  25.803,  24.676,  26.687,  26.613,
        ...      25.173,  27.17 ,  26.285,  26.826,  30.991,  28.051,  27.569,
        ...      27.075,  28.119,  28.615,  28.329,  28.289,  29.284,  28.582,
        ...      28.84 ,  28.993,  29.449,  28.919,  29.532,  29.569,  29.62 ,
        ...      30.36 ,  29.944,  31.596,  31.59 ,  30.285,  30.541,  32.068,
        ...      30.502,  30.576,  31.01 ,  32.364,  30.888,  33.179,  30.968,
        ...      30.885,  31.365,  31.257,  31.146,  31.599,  32.515,  32.316,
        ...      32.271,  32.705,  34.092,  32.716,  34.131,  32.947,  34.338,
        ...      34.427,  34.918,  35.481,  35.568,  35.213,  35.786,  35.867,
        ...      35.805,  36.186,  36.493,  36.853,  36.052,  36.522,  37.187,
        ...      36.768,  36.924,  37.565,  37.341,  37.865,  38.578,  37.872,
        ...      38.716,  38.145,  38.839,  40.927,  40.109,  40.054,  40.948,
        ...      41.086,  41.782,  41.533,  41.332,  41.701,  42.084,  42.202,
        ...      42.748,  42.857,  43.171,  43.412,  43.884,  44.064,  44.122,
        ...      44.647,  44.888,  44.878,  44.929,  45.496,  45.912,  46.018,
        ...      46.855,  46.945,  46.489,  47.46 ,  47.391,  48.26 ,  51.952,
        ...      52.749,  53.691,  55.061,  55.656])
        >>> s_chi = np.array(
        ...    [ -3.983,  -1.18 ,  -4.09 ,  -1.389,  20.68 ,   8.594,  16.175,
        ...      16.005,  20.645,  14.423, -28.116,  12.645, -30.944, -30.063,
        ...      14.302, -31.013,  22.167,  15.056, -43.934, -25.464, -18.599,
        ...       0.706,  20.823, -25.887,  18.8  ,   7.751,  19.836,  -4.655,
        ...      11.551, -11.033,  17.782,   2.158,  -5.937,  -1.77 ,   1.003,
        ...      18.599,   0.967,  29.994, -26.815,  -6.674,  12.576,  30.234,
        ...      -5.301,  -7.763,  15.665,  30.201,  -7.682, -35.206,   1.735,
        ...      -3.312,  12.502,  -7.999,   1.997,  -3.489,  15.754, -10.219,
        ...      -5.675,  13.657,  29.346,  -5.629,  28.173,   7.485, -15.096,
        ...     -11.839,  16.36 , -21.932, -23.035,  15.824,  16.383,  17.302,
        ...      15.608, -23.142, -21.983, -25.936,  -0.877,   4.743,  24.124,
        ...      -2.843,  -4.011, -22.061,  -8.448,  17.065, -29.782,   2.977,
        ...      25.746,  13.396,  -6.73 , -39.191,  12.943,  -0.268,  15.37 ,
        ...      15.709,  36.166,  20.751,  13.482,  11.316, -17.798,  13.5  ,
        ...       0.247,  11.702,  13.205,  -2.511, -37.506, -15.404, -14.046,
        ...      14.106, -20.632, -33.049, -14.697,  26.14 , -29.167,  -2.601,
        ...     -12.413,  12.397,  49.108,  23.141, -28.3  ,  -1.048,  -4.872,
        ...      -3.471,  10.512, -24.333, -19.858])
        >>> predicted_hkl, _ = pred(np.array([.5*s_tth, s_chi]))
        >>> spot1_ind, spot2_ind = 76, 54
        >>> pred.generate_orientation(s_tth, s_chi, predicted_hkl, spot1_ind, spot2_ind)
        >>>
        """
        # spots_in_center = np.arange(0, len(s_tth))
        import LaueTools.generaltools as GT
        import LaueTools.CrystalParameters as CP
        import LaueTools.findorient as FindO
        from LaueTools.matchingrate import Angular_residues_np
        import LaueTools.dict_LaueTools as dictLT

        dist = GT.calculdist_from_thetachi(np.array([.5*s_tth, s_chi]).T,
                np.array([.5*s_tth, s_chi]).T)

        dist_ = dist[spot1_ind, spot2_ind]

        lattice_params = dictLT.dict_Materials[self.material][1]
        B = CP.calc_B_RR(lattice_params)

        Gstar_metric = CP.Gstar_from_directlatticeparams(lattice_params[0],lattice_params[1],\
                                                         lattice_params[2],lattice_params[3],\
                                                         lattice_params[4],lattice_params[5])

        ## list of equivalent HKL
        hkl1_list = self.hkl_all_class[tuple(predicted_hkl[spot1_ind])]
        hkl2_list = self.hkl_all_class[tuple(predicted_hkl[spot2_ind])]
        tth_chi_spot1 = np.array([s_tth[spot1_ind], s_chi[spot1_ind]])
        tth_chi_spot2 = np.array([s_tth[spot2_ind], s_chi[spot2_ind]])
        ## generate LUT to remove possibilities of HKL
        hkl_all = np.vstack((hkl1_list, hkl2_list))
        LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)
        hkls = FindO.PlanePairs_2(dist_, 0.5, LUT, onlyclosest=1)

        if np.all(hkls == None):
            print("Nothing found")
            return np.zeros((3,3)), 0.0

        rot_mat = []
        mr = []
        for ii in range(len(hkls)):
        # ii = 0
            rot_mat1 = FindO.OrientMatrix_from_2hkl(hkls[ii][0], tth_chi_spot1, \
                                                    hkls[ii][1], tth_chi_spot2,
                                                    B)

            AngRes = Angular_residues_np(rot_mat1, s_tth, s_chi,
                                                key_material=self.material,
                                                emax=emax,
                                                ResolutionAngstrom=False,
                                                ang_tol=0.5,
                                                detectorparameters=self.dict_dp,
                                                dictmaterials=dictLT.dict_Materials)
            allres, _, nbclose, nballres, _, _ = AngRes
            match_rate = nbclose/nballres
            rot_mat.append(rot_mat1)
            mr.append(match_rate)

        max_ind = np.argmax(mr)

        return rot_mat[max_ind], mr[max_ind]
