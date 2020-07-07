# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:11:10 2019

@author: ZML
"""

from .resnet import get_wide_resnet50_2, get_wide_resnet101_2, get_resnet18

from .alexnet import get_alexnet

from .simple_model import get_Net

from .vgg import get_vgg19_bn

#import resnet, alexnet, simple_model



def get_model(model='simple_net',num_classes=10):
    model = model.strip().upper()
    if model=='SIMPLE_NET':
        return get_Net(num_classes)
    elif model == 'WIDE_RESNET50':
        return get_wide_resnet50_2(pretrained=False, num_class=num_classes)
    elif model == 'WIDE_RESNET101':
        return get_wide_resnet101_2(pretrained=False, num_class=num_classes)
    elif model == 'RESNET18':
        return get_resnet18(pretrained=False, num_class=num_classes)
    elif model == 'ALEXNET':
        return get_alexnet(pretrained=True, num_classes=num_classes)
    elif model == 'VGG19_BN':
        return get_vgg19_bn(pretrained=False, num_classes=num_classes)    
    else:
        assert 1==0,'please select valid model'
    
def model_to_model(pretrained_model, raw_model):
    
    pretrained_model_dict=pretrained_model.state_dict()
    raw_model_dict=raw_model.state_dict()
    parameters = {k:v for k, v in pretrained_model_dict.items() if k in raw_model_dict}
    raw_model_dict.update(parameters)
    
    return raw_model.load_dict(raw_model_dict)