import unittest

from libics.cfg import io


###############################################################################


class NamedTupleTestCase(unittest.TestCase):

    def setUp(self):
        self.dict = {"a": 1, "b": "B", "c": 4.12}
        self.dict_nest = {"a": 1, "b": "B", "c": 4.12,
                          "d": {"z": "Z", "yy": "Y", "x": -3}}

    def test_nested_namedtuple(self):
        io.NamedTuple(self.dict)
        io.NamedTuple(self.dict_nest)


###############################################################################


if __name__ == '__main__':
    unittest.main()
