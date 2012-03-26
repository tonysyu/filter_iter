===============================================
Proof of concept for generic filter development
===============================================

:author: Nadav Horesh

A generic filter platform to overcome deficiencies of `ndimage.generic_filter`
and `numpy`'s neighborhood iterator. In particular, this is supposed to be
easier to use and enable multi-threaded and faster execution. With this package
the user can write processing kernel functions in python for the early research
phase, and later rewrite the code in pure C (without any boiler-plate code).
The user's C kernel is called in a loop without any python in the middle.

The code here is just a proof-of-concept: it needs a major extension
and rewrite.

Files
=====

filters_iter.pyx: Low level
gen_filter.py: A high level


Description
===========

filter_iter.pyx
---------------

The module contains the class `gen_filter_matrix` which is initialization
with a 2D array to be filtered, and the parameters `kernel_shape`,
`padding`, `cval` which correspond to ndimage.generic_filter's `size`, `mode`
and `cval` parameters. The class provides methods to apply either a
python or C call-back function.


gen_filter.py
-------------

A high level module to make it easy to use a C call-back. The module contains
a function `apply_C_filter`, which expects one-or-more compiled C functions.

For some reason (that could be my limited skill) the function can
access the call-back function via a pointer to that function defined in
the same dll::

   float my_filter(float* buf, int size)
   {
     ...
   }

The following line is a must in order to call the function::

   void* my_filterp = my_filterp

The name "my_filterp" must be provided as the `c_function` parameter to
`apply_C_filter`.

