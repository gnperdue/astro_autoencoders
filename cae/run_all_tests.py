'''
run all the tests
'''
import unittest
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='test', type=str,
                    help='pattern base name')
parser.add_argument('--verbosity', default=2, type=int,
                    help='test verbosity (int)')


def main(pattern, verbosity):
    pattern = pattern + '*.py'
    suite = unittest.TestLoader().discover('./tests/', pattern=pattern)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
