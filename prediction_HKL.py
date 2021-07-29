#!/usr/bin/env python
# coding: utf-8
# # Deep learning for HKL classification# 
# ## Now can predict with > 95% accuracy for multi grain Laue Patterns
# ## If you have model save files; go to cell 45 to load and start prediction
# ## Pros: Impressive speed for prediction; results not dependent on the statistical descriptor (ex: correlation function or distance measurment)
# ## Cons: Building reliable test data can take few hours (this will significantly increase for less symmetry crystals) --> multiprocessing to reduce time
# ## Library import
# In[82]:
import numpy as np
import _pickle as cPickle
## LaueTools import
import LaueTools.dict_LaueTools as dictLT
import LaueTools.generaltools as GT
import LaueTools.CrystalParameters as CP
from LaueTools.matchingrate import Angular_residues_np
import LaueTools.findorient as FindO

## External libraries
## for faster binning of histogram
## C version of hist
from fast_histogram import histogram1d
import h5py

class predictionHKL:
    
    def __init__(self, input_params, directory=None, CCDLabel="MARCCD165"):
        framedim = dictLT.dict_CCD[CCDLabel][0]
        self.dict_dp={}
        self.dict_dp['kf_direction']='Z>0'
        self.dict_dp['detectorparameters'] = input_params["detectorparameters"]
        self.dict_dp['detectordistance'] = input_params["detectorparameters"][0]
        self.dict_dp['detectordiameter'] = input_params["pixelsize"]*framedim[0]
        self.dict_dp['pixelsize'] = input_params["pixelsize"]
        self.dict_dp['dim'] = framedim
        self.material_ = input_params["material_"]
        
        self.save_directory = directory
        load_weights = self.save_directory+"//my_model_"+self.material_+".h5"
        
        with open(self.save_directory+"//classhkl_data_"+self.material_+".pickle", "rb") as input_file:
            _, _, _, _, _, self.hkl_all_class, _, _, _ = cPickle.load(input_file)
            
        # =============================================================================
        # ## for a fixed architecture
        # ## rebuild if architecture changes
        # =============================================================================
        self.wb = self.read_hdf5(load_weights)
        self.temp_key = list(self.wb.keys())
        
        self.classhkl = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
        self.angbins = np.load(self.save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
        
    # ## Load model weights for prediction
    #% Numpy load model
    def read_hdf5(self, path):
        weights = {}
        keys = []
        with h5py.File(path, 'r') as f: # open file
            f.visit(keys.append) # append all keys to list
            for key in keys:
                if ':' in key: # contains data if ':' in key
                    weights[f[key].name] = f[key].value
        return weights
    
    def softmax(self, x):
        return (np.exp(x).T / np.sum(np.exp(x).T, axis=0)).T
        
    def predict(self, x):
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
        l3 = self.softmax(l3) ## output softmax activation
        return l3
    
    def predict_HKL(self, data_theta, data_chi):

        sorted_data = np.transpose(np.array([data_theta, data_chi]))
        tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))
        
        codebars_all = []        
        spots_in_center = np.arange(0,len(data_theta))

        for i in spots_in_center:
            spotangles = tabledistancerandom[i]
            spotangles = np.delete(spotangles, i)# removing the self distance
            # codebars = np.histogram(spotangles, bins=angbins)[0]
            codebars = histogram1d(spotangles, range=[min(self.angbins),max(self.angbins)], bins=len(self.angbins)-1)
            ## normalize the same way as training data
            max_codebars = np.max(codebars)
            codebars = codebars/ max_codebars
            codebars_all.append(codebars)
        ## reshape for the model to predict all spots at once
        codebars = np.array(codebars_all)
        ## Do prediction of all spots at once
        prediction = self.predict(codebars)
        max_pred = np.max(prediction, axis = 1)
        class_predicted = np.argmax(prediction, axis = 1)
        predicted_hkl = self.classhkl[class_predicted]
        ## return predicted HKL and their softmax confidence
        return predicted_hkl, max_pred
    
    def generate_orientation(self, s_tth, s_chi, predicted_hkl, spot1_ind, spot2_ind, emax=23, ):
        
        spots_in_center = np.arange(0,len(s_tth))
        
        dist = GT.calculdist_from_thetachi(np.array([s_tth[spots_in_center]/2., s_chi[spots_in_center]]).T,
                                                        np.array([s_tth[spots_in_center]/2., s_chi[spots_in_center]]).T)
        
        dist_ = dist[spot1_ind, spot2_ind]
        
        lattice_params = dictLT.dict_Materials[self.material_][1]
        B = CP.calc_B_RR(lattice_params)
        
        Gstar_metric = CP.Gstar_from_directlatticeparams(lattice_params[0],lattice_params[1],\
                                                         lattice_params[2],lattice_params[3],\
                                                             lattice_params[4],lattice_params[5])

        ## list of equivalent HKL
        hkl1 = self.hkl_all_class[str(predicted_hkl[spot1_ind])]
        hkl1_list = np.array([ii.miller_indices() for ii in hkl1["family"]])
        hkl2 = self.hkl_all_class[str(predicted_hkl[spot2_ind])]
        hkl2_list = np.array([ii.miller_indices() for ii in hkl2["family"]])
        tth_chi_spot1 = np.array([s_tth[spots_in_center[spot1_ind]], s_chi[spots_in_center[spot1_ind]]])
        tth_chi_spot2 = np.array([s_tth[spots_in_center[spot2_ind]], s_chi[spots_in_center[spot2_ind]]])
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
                                                key_material=self.material_,
                                                emax=emax,
                                                ResolutionAngstrom=False,
                                                ang_tol=0.5,
                                                detectorparameters=self.dict_dp,
                                                dictmaterials=dictLT.dict_Materials)  
            
            (allres, _, nbclose, nballres, _, _) = AngRes
            match_rate = nbclose/nballres
            rot_mat.append(rot_mat1)
            mr.append(match_rate)
            
        max_ind = np.argmax(mr)

        return rot_mat[max_ind], mr[max_ind]
    