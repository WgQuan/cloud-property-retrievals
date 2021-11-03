#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' This is the code for predicting'

__author__ = 'Quan Wang'

import numpy as np
import torch

from model import UNet_MTL

class Predictor(object):
    def __init__(self, Input, Norm_Params, Model_Path):
        self.input = torch.from_numpy(Input).unsqueeze(0).float()
        self.norm_params = Norm_Params

        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

        self.model = UNet_MTL(input_channels=8, num_classes=3).to(self.device)
        
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(Model_Path)           
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


    def testing(self):
        print('Now is testing ...')

        ### model evaluation
        self.model.eval()
        with torch.no_grad():
            inputs = self.input.to(device=self.device, non_blocking = True)

            outputs = self.model(inputs)

            COT_out =outputs[0,0,:,:]
            CER_out =outputs[0,1,:,:]
            CTH_out =outputs[0,2,:,:]

            CNN_COT = COT_out.detach().cpu().numpy()
            CNN_CER = CER_out.detach().cpu().numpy()
            CNN_CTH = CTH_out.detach().cpu().numpy()

            CNN_COT = CNN_COT*self.norm_params['COT_Std'][0] + self.norm_params['COT_Mean'][0]
            CNN_CER = CNN_CER*self.norm_params['CER_Std'][0] + self.norm_params['CER_Mean'][0]
            CNN_CTH = (CNN_CTH*self.norm_params['CTH_Std'][0] + self.norm_params['CTH_Mean'][0])/1000

            dst_data = {'CNN_COT':CNN_COT, 'CNN_CER':CNN_CER, 'CNN_CTH':CNN_CTH}

        return dst_data