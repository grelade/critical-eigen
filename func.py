import os
import sys
import numpy as np

def determine_unique_postfix(fn) -> str:
    """
    Determine the unique postfix for a file or directory in order to avoid overwriting
    directories created during the run.
    """
    if not os.path.exists(fn):
        return ""
    path, name = os.path.split(fn)
    name, ext = os.path.splitext(name)
    make_fn = lambda i: os.path.join(path, "{}_{}{}".format(name, i, ext))
    for i in range(1, sys.maxsize):
        uni_fn = make_fn(i)
        if not os.path.exists(uni_fn):
            return "_" + str(i)
        
        

def concatenate_dict(dictionary: dict,axis: int = 0) -> np.array:
    '''
    concatenate dictionary of np.arrays along an axis
    '''
    data_batch = None
    for key,data in dictionary.items():
        data = np.expand_dims(data,axis=axis)
        data_batch = data if data_batch is None else np.concatenate((data_batch,data),axis=axis)
        
    return data_batch
        
def batch_calc(data_batch: np.array,func,axis: int = 0) -> np.array:
    '''
    Run func over data_batch np.array and cocnatenate the output along an axis
    '''
    outputs = None
    for datum in data_batch:
        output = func(datum)
        output = np.expand_dims(output,axis=axis)
        outputs = output if outputs is None else np.concatenate((outputs,output),axis=axis)
        
    return outputs