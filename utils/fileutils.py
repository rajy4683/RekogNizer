"""
    This file contains non-core utility functions 
    used in the overall project
"""

import random
import os
import numpy as np
import pandas as pd


def rand_run_name():
    ran = random.randrange(10**80)
    myhex = "%064x" % ran
    #limit string to 64 characters
    myhex = myhex[:10]
    return myhex

def generate_model_save_path(base="/content/drive/My Drive/EVA4/model_saves",rand_string=None):
    if rand_string == None:
        rand_string=rand_run_name()
    file_name = "model-"+rand_string+".h5"
    return os.path.join(base,file_name)