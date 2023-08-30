from argparse import ArgumentParser
import logging

from utils.log import setup_logging
from utils.run_config import RunConfig
from utils.cli_frontend import add_cloudsc_size_arguments
from execute.parameters import ParametersProvider
from measurements.profile_config import ProfileConfig

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("programs", help="Name of the program to run", nargs='+')
    parser.add_argument('--k-caching', action='store_true', default=False, help="use k-caching")
    parser.add_argument('--change-stride', action='store_true', default=False, help="change stride")
    parser.add_argument('--log-level', default='info')
    add_cloudsc_size_arguments(parser)
    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())

    for program in args.programs:
        run_config = RunConfig()
        run_config.set_from_args(args)
        params = ParametersProvider(program)
        params.update_from_args(args)
        profile_config = ProfileConfig(program, [params], ['NBLOCKS'], ncu_repetitions=1, tot_time_repetitions=2,
                use_basic_sdfg=True)
        profile_config.profile(run_config)


if __name__ == '__main__':
    main()
