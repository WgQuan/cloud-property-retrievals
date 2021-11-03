#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' This is the code to reading MYD02 and MYD03 .hdf file'

__author__ = 'Quan Wang'

import numpy as np
from pyhdf.SD import SDC, SD

class HDF_READ(object):

    ### MYD021KM**************************************
    def Read_MYD02(self, File, Keywords, Factor = 'radiance'):
        all_data = []

        MYD02_hdf = SD(File, SDC.READ)

        for idx, filed_name in enumerate(Keywords):
            MYD02 = MYD02_hdf.select(filed_name)
            data = MYD02.get().astype(np.float)

            attrs = MYD02.attributes()

            invalid = np.full(data.shape, False)
            if "valid_range" in attrs.keys():
                vra = attrs["valid_range"]
                valid_min = vra[0]
                valid_max = vra[1]
                invalid = np.logical_or(data > valid_max,
                                        data < valid_min)
            if "_FillValue" in attrs.keys():
                _FillValue = attrs["_FillValue"]
                invalid = np.logical_or(invalid, data == _FillValue)

            data[invalid] = np.nan

            if filed_name == 'EV_1KM_Emissive':
                offsets = np.array(attrs["%s_offsets" % Factor])
                scale = np.array(attrs["%s_scales" % Factor])

                offsets = offsets.reshape(-1,1,1)
                scale = scale.reshape(-1,1,1)

                data = (data - offsets) * scale
                data = np.nan_to_num(data)

            all_data.append(data)
        
        MYD02_hdf.end()  
        return all_data

    ### MYD03******************************************
    def Read_MYD03(self, File, Keywords):
        all_data = []

        MYD03_hdf = SD(File, SDC.READ)

        for idx, filed_name in enumerate(Keywords):
            MYD03 = MYD03_hdf.select(filed_name)
            data = MYD03.get().astype(np.float)

            attrs = MYD03.attributes()

            invalid = np.full(data.shape, False)
            if "valid_range" in attrs.keys():
                vra = attrs["valid_range"]
                valid_min = vra[0]
                valid_max = vra[1]
                invalid = np.logical_or(data > valid_max,
                                        data < valid_min)
            if "_FillValue" in attrs.keys():
                _FillValue = attrs["_FillValue"]
                invalid = np.logical_or(invalid, data == _FillValue)

            data[invalid] = np.nan

            if "scale_factor" in attrs.keys():
                scale_factor = attrs['scale_factor']
                data = data * scale_factor

            data = np.nan_to_num(data)

            all_data.append(data)

        MYD03_hdf.end()
        return all_data