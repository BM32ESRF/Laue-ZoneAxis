#!/usr/bin/env python3
import numpy as np
import LaueTools.dict_LaueTools as DictLT
import LaueTools.IOLaueTools as IOLT
import LaueTools.LaueGeometry as Lgeo
from prediction_HKL import predictionHKL
# from prediction_as_function import prediction_rm
import glob, re
import laue

rectpix = DictLT.RECTPIX  # see above  camera skewness
PI = np.pi
DEG = PI / 180.0

## detector parameters
detectorplaneparameters = [70.22, 1039.395, 943.57, 0.7478, 0.07186] #ZBB1
pixelsize = 0.079142
CCDLabel="MARCCD165"

## parameters for NN prediction
input_params = {
                "material_": "ZrO2", ## same key as used in LaueTools
                "detectorparameters" : detectorplaneparameters,
                "pixelsize" : pixelsize,
                "emax":21,
                "tolerance":0.5
                } 
model_direc = r"C:\Users\purushot\Desktop\pattern_matching\experimental\misc\prediction_from_file\ZrO2"

## directory details
sample_name = "ZBB1_ROI1"
rep = r"C:\Users\purushot\Desktop\bm32_data"+"\\"+sample_name
format_file = "mccd"

list_of_files = glob.glob(rep+'\\*.'+format_file)
## sort files
list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

if __name__ == "__main__":
    
    # all_stats = []
    for ind, files in enumerate(list_of_files):
        if ind >= 1:
            continue
        ## Robin's peak search
        diagram = laue.Experiment(files, threshold=3.5, dd=detectorplaneparameters[0], \
                                                        xcen=detectorplaneparameters[1],\
                                                        ycen= detectorplaneparameters[2], \
                                                        bet=detectorplaneparameters[3], \
                                                        gam=detectorplaneparameters[4])[0]
            
        pics_x, pics_y = diagram.get_positions()
        intensity_peaks = np.ones(len(pics_x))
        
        zoneaxis = diagram.find_zone_axes()
        
        ## Get subsets
        
        ## Sort the indices by prediction confidence
        
        ## convert Peak XY to 2theta, chi with LaueTools
        twicetheta, chi = Lgeo.calc_uflab(pics_x, pics_y, detectorplaneparameters,
                                            returnAngles=1,
                                            pixelsize=pixelsize,
                                            kf_direction='Z>0')
        
        data_theta, data_chi = twicetheta/2., chi    

        ## Neural network prediction routine below
        ## need MOD_grain_classhkl_angbin.pickle
        ## classhkl_data_{material}.pickle
        ## my_model_{material}.pickle
        model = predictionHKL(input_params, directory=model_direc, CCDLabel=CCDLabel)
        # model = prediction_rm(input_params)
        ## just HKL predictions
        hkl_list, pred_max = model.predict_HKL(data_theta, data_chi)
        
# =============================================================================
#         
# =============================================================================
        spot1_ind, spot2_ind = 464, 472## output of zone axis
        
        
        
        rot_matrix, match_rate = model.generate_orientation(twicetheta, chi, hkl_list, spot1_ind, spot2_ind, emax=input_params["emax"])
        
        # stats = model.prediction_(input_params, data_theta, data_chi)
        # all_stats.append(stats)
        
        ## write COR file to be opened with LaueTools to verify
        ## Maybe write dat files instead of COR
        CCDcalib = {"CCDLabel":CCDLabel,
                    "dd":detectorplaneparameters[0], 
                    "xcen":detectorplaneparameters[1], 
                    "ycen":detectorplaneparameters[2], 
                    "xbet":detectorplaneparameters[3], 
                    "xgam":detectorplaneparameters[4],
                    "pixelsize": pixelsize}
        
        IOLT.writefile_cor("grain_"+str(ind), twicetheta, chi, pics_x, pics_y, intensity_peaks,
                           param=CCDcalib, sortedexit=0)
