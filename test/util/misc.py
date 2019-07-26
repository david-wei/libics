import unittest

from libics.util import misc


###############################################################################


class StringFunctionsTestCase(unittest.TestCase):

    def test_char_range(self):
        self.assertEqual(list(misc.char_range("d")), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.char_range(4)), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.char_range("b", "d")), ["b", "c", "d"])
        self.assertEqual(list(misc.char_range("b", 3)), ["b", "c", "d"])
        self.assertEqual(list(misc.char_range("a", "d", 2)), ["a", "c"])
        self.assertEqual(list(misc.char_range("e", "a")), [])


###############################################################################


if __name__ == '__main__':
    unittest.main()
