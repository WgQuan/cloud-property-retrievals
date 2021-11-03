#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' This is the code of utility'

__author__ = 'Quan Wang'

import numpy as np
import h5py

from HDF_Read import HDF_READ

class Data_Process(object):
    def __init__(self, Norm_File):
        super(Data_Process, self).__init__()
        self.norm_file = Norm_File
        self.norm_params = self.Read_Normfile(self.norm_file)

    def Read_Normfile(self, File):
        dataset = h5py.File(File, 'r')
        params = {k:np.array(dataset[k]) for k in dataset.keys()}
        dataset.close()

        return params
    
    def Read_Input(self, MYD02_File, MYD03_File,):
        READ = HDF_READ()
        myd02_data = READ.Read_MYD02(MYD02_File, ['EV_1KM_Emissive'], Factor = 'radiance')[0]
        sensor_angles = np.array(READ.Read_MYD03(MYD03_File, ['SensorZenith']))
        altitude = np.array(READ.Read_MYD03(MYD03_File, ['Height']))

        emis_norm = self.Norm_Data(myd02_data, self.norm_params, 'Emissive', [8,10,11,12,13,14])
        vza_norm = self.Norm_Data(sensor_angles, self.norm_params, 'SensorAngles', [0])
        alt_norm = self.Norm_Data(altitude, self.norm_params, 'Altitude', [0])

        input = np.concatenate([emis_norm, vza_norm, alt_norm])

        return input

    def Norm_Data(self, Data, Norm_Params, Feature, Idxs):
        data = Data[Idxs]

        if 'Angle' in Feature:
            data = np.cos(Data*np.pi/180)
        data = np.nan_to_num(data)

        _mean = Norm_Params['%s_Mean' % Feature][Idxs]
        _std = Norm_Params['%s_Std' % Feature][Idxs]

        data = (data - _mean)/_std

        if (len(Idxs) == 1) & (len(data.shape) == 2):
            data = np.expand_dims(data, 0)

        return data
    
    def Save_Data(self, File, Key, Data):
        dataset = h5py.File(File, 'a')
        if Key not in dataset.keys():
            dataset.create_dataset(Key, data=Data)
            dataset.close()
        else:
            del dataset[Key]
            dataset.close()

            self.Save_Data(File, Key, Data)
