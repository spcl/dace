from argparse import ArgumentParser


class Script:
    name: str
    description: str

    def __init__(self, subparsers):
        parser = subparsers.add_parser(self.name, help=self.description)
        parser.set_defaults(func=self.action)
        self.add_args(parser)

    @staticmethod
    def action(args):
        raise NotImplementedError

    def add_args(self, parser: ArgumentParser):
        pass
