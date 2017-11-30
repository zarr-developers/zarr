# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import codecs


import numpy as np


def parse_vlen_dtype(dtype):
    vlen_setup = dtype.split(':')

    # check vlen type is not missing
    if len(vlen_setup) < 2:
        # TODO improve error message
        raise ValueError('invalid vlen dtype, missing item type')
    vlen_type = vlen_setup[1]

    if vlen_type == 'bytes':
        metadata = dict(vlen=vlen_type)

    elif vlen_type == 'text':
        # determine text encoding
        if len(vlen_setup) == 2:
            encoding = 'utf-8'
        else:
            encoding = vlen_setup[2]
        # normalize encoding name
        encoding = codecs.lookup(encoding).name
        metadata = dict(vlen=vlen_type, encoding=encoding)

    else:
        # normalize primitive type representation
        vlen_type = np.dtype(vlen_type).str
        metadata = dict(vlen=vlen_type)

    dtype = np.dtype(object, metadata=metadata)
    return dtype
