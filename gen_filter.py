"""
Created on Sun Mar 11 10:32:31 2012

High level interface to filter_iter
@author: nadav
"""
from __future__ import print_function

import numpy as np
import ctypes
import _filter_iter as fiter
from skimage.util.dtype import _convert


def apply_C_filter(image, dll, c_function, filter_size, mode='constant', cval=0):
    """
    Apply a C function based filter on image

    Parameters
    ----------
    image : 2D array
        Input image.
    dll: str or a ctypes dll object
        If str: A dll name (including full path if not on the default search
        path).
    c_function: str or int
        If str: The name of the variable in the dll pointing to the function.
        If int: A Function pointer.
    filter_size : (2,) array
        Filter shape.
    mode : str
        Padding mode.
    cval : a scalar
        Padding fill value (applicable is mode == 'const')

    Returns
    -------
    output : array
        A 2D array of the same dtype and shape as the input array.
    """
    # A temporary confinment to float32
    image = _convert(image, np.float32)
    if type(dll) is str:
        dll = ctypes.cdll.LoadLibrary(dll)
    if type(c_function) is str:
        pfcn =  ctypes.c_voidp.in_dll(dll, c_function).value
    else:
        pfcn = c_function
    # Prepare paded data
    padded = fiter.gen_filter_matrix(image, filter_size, mode=mode, cval=cval)
    output = np.empty_like(image)
    padded.filter_with_C_callback_float(pfcn, output)
    return output

