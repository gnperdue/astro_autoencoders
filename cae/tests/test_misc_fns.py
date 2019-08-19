'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import unittest
import logging

import tfcae.misc_fns as misc_fns


class TestUtils(unittest.TestCase):

    def test_get_logging_level(self):
        self.assertEqual(logging.INFO,
                         misc_fns.get_logging_level('info'))
        self.assertEqual(logging.INFO,
                         misc_fns.get_logging_level('INFO'))
        self.assertEqual(logging.DEBUG,
                         misc_fns.get_logging_level('debug'))
        self.assertEqual(logging.DEBUG,
                         misc_fns.get_logging_level('DEBUG'))
        self.assertEqual(logging.WARNING,
                         misc_fns.get_logging_level('warning'))
        self.assertEqual(logging.WARNING,
                         misc_fns.get_logging_level('WARNING'))
        self.assertEqual(logging.ERROR,
                         misc_fns.get_logging_level('error'))
        self.assertEqual(logging.ERROR,
                         misc_fns.get_logging_level('ERROR'))
        self.assertEqual(logging.CRITICAL,
                         misc_fns.get_logging_level('critical'))
        self.assertEqual(logging.CRITICAL,
                         misc_fns.get_logging_level('CRITICAL'))
        self.assertEqual(logging.INFO,
                         misc_fns.get_logging_level('NoSuchLevel'))


if __name__ == '__main__':
    unittest.main()
