import unittest
from zarr.creation import array


class TestArrayMissingKeys(unittest.TestCase):
    def test_raises_on_missing_key(self):
        a = array(range(4), chunks=2)

        # configure raise on missing chunk
        a.set_options(fill_missing_chunk=False)

        # pop first chunk
        a.chunk_store.pop("0")

        # read from avaible chunk w/o error
        b = a[-1]
        c = a[-2:]

        # reading missing chunk should raise
        with self.assertRaises(KeyError):
            b = a[0]

        with self.assertRaises(KeyError):
            c = a[:2]
