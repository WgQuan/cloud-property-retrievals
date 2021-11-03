#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'This is the main part for the prediction'

__author__ = 'Quan Wang'

import os
from Predict import Predictor
from utils import Data_Process

def main(Norm_File, Model_File, MYD02_File, MYD03_File, Dst_File):
    DATA_PROCESS = Data_Process(Norm_File)
    norm_params = DATA_PROCESS.Read_Normfile(Norm_File)
    input = DATA_PROCESS.Read_Input(MYD02_File, MYD03_File)

    predictor = Predictor(input, norm_params, Model_File)
    data = predictor.testing()

    for k, v in data.items():
        DATA_PROCESS.Save_Data(Dst_File, k, v)

if __name__ == '__main__':
    module_path = os.path.dirname(__file__)
    os.chdir(module_path)

    Norm_file = '../Normalizition_Parameters/Normalization_Parameters_2010.h5'
    MYD02_file = '../Test_Data/MYD021KM.A2009001.1255.061.2018040094617.hdf'
    MYD03_file = '../Test_Data/MYD03.A2009001.1255.061.2018039204219.hdf'
    Model_file = '../Model/model.pkl'
    Dst_file = '../Test_Data/Dst.h5'

    main(Norm_file, Model_file, MYD02_file, MYD03_file, Dst_file)
