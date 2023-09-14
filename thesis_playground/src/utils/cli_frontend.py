from argparse import ArgumentParser


def add_cloudsc_size_arguments(parser: ArgumentParser):
    """
    Adds arguments to the given ArgumentParser controlling the sizes of the arrays used in cloudsc.

    :param parser: The argument parser to add the arguments to
    :type parser: ArgumentParser
    """
    parser.add_argument('--KLON', type=int, default=None)
    parser.add_argument('--KLEV', type=int, default=None)
    parser.add_argument('--NBLOCKS', type=int, default=None)
    parser.add_argument('--KIDIA', type=int, default=None)
    parser.add_argument('--KFDIA', type=int, default=None)
    parser.add_argument('--NCLV', type=int, default=None)
