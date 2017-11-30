# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import codecs


import numpy as np


def is_vlen_dtype(dtype):
    return dtype == object and dtype.metadata and 'vlen' in dtype.metadata


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
        vlen_type = np.dtype(vlen_type)
        assert vlen_type != object  # TODO value error with message
        metadata = dict(vlen=vlen_type)

    dtype = np.dtype(object, metadata=metadata)
    return dtype


def serialize_vlen_dtype(dtype):
    vlen_type = dtype.metadata['vlen']
    if isinstance(vlen_type, np.dtype):
        vlen_type = vlen_type.str
    s = 'vlen:{}'.format(vlen_type)
    if vlen_type == 'text':
        vlen_encoding = dtype.metadata['encoding']
        s += ':{}'.format(vlen_encoding)
    return s


# noinspection PyMethodMayBeStatic
class VlenBytesCodec(object):

    def encode(self, values):
        print('VlenBytesCodec.encode', values.dtype, values)

        # sanity checks
        assert isinstance(values, np.ndarray)
        assert values.dtype == object

        # number of items
        n = np.array(values.size, dtype='<i8')

        # more sanity checks
        if n > 0:
            assert isinstance(values[0], bytes), values[0]

        # compute lengths
        lengths = np.fromiter((len(v) for v in values.flat), dtype='<i4', count=n)

        # concatenate values, assume all items are bytes already
        data = b''.join(values.flat)

        # build final buffer
        enc = b''.join([n, lengths, data])

        return enc

    def decode(self, enc):
        print('VlenBytesCodec.decode; enc', enc)

        # extract number of items
        assert len(enc) >= 8  # sanity check
        n = np.frombuffer(enc[:8], dtype='<i8')[0]

        # extract lengths
        enc_lengths = enc[8:8+(n*4)]
        assert len(enc_lengths) == n * 4  # sanity check
        lengths = np.frombuffer(enc_lengths, dtype='<i4')

        # compute offsets
        offsets = np.zeros(n + 1, dtype='i4')
        np.cumsum(lengths, out=offsets[1:])

        # build values
        len_data = offsets[-1]
        data = enc[8+(n*4):]
        assert len(data) == len_data  # sanity check
        values = [bytes(data[i:j]) for i, j in zip(offsets[:-1], offsets[1:])]
        values = np.array(values, dtype=object)

        print('VlenBytesCodec.decode; values', values)
        return values
