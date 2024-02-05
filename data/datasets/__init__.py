# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .dataset_loader import ImageDataset
from .adp_cuhk03 import adpCUHK03
from .adp_all import adpALL

# this dict contains the names of the availabe datasets as keys,
# and as valutes the respective classes to initialize them
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'cuhk03': adpCUHK03,
    'all': adpALL
}


# get available datasets list
def get_names():
    return __factory.keys()


# get an instance of a given dataset
def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
