# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import os
import platform
import tempfile
import yaml
import warnings


@contextlib.contextmanager
def set_temporary(*path, value):
    """ Temporarily set configuration value at ``path`` to value, and reset it after the context manager exits.

        :Example:

            print(Config.get("compiler", "build_type")
            with set_temporary("compiler", "build_type", value="Debug"):
                print(Config.get("compiler", "build_type")
            print(Config.get("compiler", "build_type")

    """
    old_value = Config.get(*path)
    Config.set(*path, value=value)
    yield
    Config.set(*path, value=old_value)


@contextlib.contextmanager
def temporary_config():
    """
    Creates a context where all configuration options changed will be reset when the context exits.

    with temporary_config():
        Config.set("testing", "serialization", value=True)
        Config.set("optimizer", "autooptimize", value=True)
        foo()
    """
    with tempfile.NamedTemporaryFile() as fp:
        Config.save(fp.name)
        yield
        Config.load(fp.name)


def _env2bool(envval):
    """ Converts an arbitrary value to boolean.
        :param envval: Arbitrary value.
        :return: True if the input value matches a valid TRUE
                  value, or False otherwise.
    """
    return str(envval).lower() in ['true', '1', 'y', 'yes', 'on', 'verbose']


def _add_defaults(config, metadata):
    """ Add defaults to configuration from metadata.
        :return: True if configuration was modified, False otherwise.
    """
    osname = platform.system()
    modified = False
    for k, v in metadata.items():
        # Recursive call for fields inside the dictionary
        if v['type'] == 'dict':
            if k not in config:
                modified = True
                config[k] = {}
            modified |= _add_defaults(config[k], v['required'])
            continue
        # Empty list initialization (if no default is specified)
        elif v['type'] == 'list':
            if k not in config and 'default' not in v:
                modified = True
                config[k] = []
                continue
        # Key does not exist in configuration, add default value
        if k not in config:
            modified = True
            # Per-OS default
            if 'default_' + osname in v:
                config[k] = v['default_' + osname]
            else:
                config[k] = v['default']
    return modified


class Config(object):
    """ Interface to the DaCe hierarchical configuration file. """

    _config = {}
    _config_metadata = {}
    _cfg_filename = None
    _metadata_filename = None

    @staticmethod
    def cfg_filename():
        """ Returns the current configuration file path. """

        return Config._cfg_filename

    @staticmethod
    def initialize():
        """
        Initializes configuration.

        :note: This function runs automatically when the module is loaded.
        """

        # If already initialized, skip
        if Config._cfg_filename is not None:
            return

        # Override default configuration file path
        default_filename = '.dace.conf'
        if 'DACE_CONFIG' in os.environ:
            default_cfg_filename = os.environ['DACE_CONFIG']
        else:
            home = os.path.expanduser("~")
            default_cfg_filename = os.path.join(home, default_filename)

        dace_path = os.path.dirname(os.path.abspath(__file__))
        Config._metadata_filename = os.path.join(dace_path, 'config_schema.yml')

        # Load configuration schema (for validation and defaults)
        Config.load_schema()

        # Priority order: current working directory, default configuration file (DACE_CONFIG), then ~/.dace.conf
        for filename in [default_filename, default_cfg_filename]:
            Config._cfg_filename = filename
            try:
                if os.path.isfile(filename):
                    Config.load()
                    break
            except (FileNotFoundError, PermissionError, OSError):
                # If any filesystem-related error happened during file load, move on to next candidate
                continue
        else:
            # None of the files were found

            # Load the defaults from metadata
            Config._config = {}
            _add_defaults(Config._config, Config._config_metadata['required'])

            # Try to create a new config file in reversed priority order, and if all else fails keep config in memory
            for filename in [default_cfg_filename, default_filename, None]:
                Config._cfg_filename = filename
                try:
                    Config.save()
                    break
                except (FileNotFoundError, PermissionError, OSError):
                    # If any filesystem-related error happened during file save, move on to next candidate
                    continue
            else:
                warnings.warn('No DaCe configuration file was able to be saved')

    @staticmethod
    def load(filename=None):
        """ Loads a configuration from an existing file.
            :param filename: The file to load. If unspecified,
                             uses default configuration file.
        """
        if filename is None:
            filename = Config._cfg_filename

        # Read configuration file
        with open(filename, 'r') as f:
            Config._config = yaml.load(f.read(), Loader=yaml.SafeLoader)

        if Config._config is None:
            Config._config = {}

        # Add defaults from metadata
        modified = _add_defaults(Config._config, Config._config_metadata['required'])
        if modified:  # Update file if changed
            Config.save()

    @staticmethod
    def load_schema(filename=None):
        """ Loads a configuration schema from an existing file.
            :param filename: The file to load. If unspecified,
                             uses default schema file.
        """
        if filename is None:
            filename = Config._metadata_filename
        with open(filename, 'r') as f:
            Config._config_metadata = yaml.load(f.read(), Loader=yaml.SafeLoader)

    @staticmethod
    def save(path=None):
        """ Saves the current configuration to a file.
            :param path: The file to save to. If unspecified,
                         uses default configuration file.
        """
        if path is None:
            path = Config._cfg_filename
        if path is not None:
            # Write configuration file
            with open(path, 'w') as f:
                yaml.dump(Config._config, f, default_flow_style=False)

    @staticmethod
    def get_metadata(*key_hierarchy):
        """ Returns the configuration specification of a given entry
            from the schema.
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration specification as a dictionary.
        """
        # Traverse the key hierarchy
        current_conf = Config._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        return current_conf

    @staticmethod
    def get_default(*key_hierarchy):
        """ Returns the default value of a given configuration entry.
            Takes into accound current operating system.
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Default configuration value.
        """
        # Traverse the key hierarchy
        current_conf = Config._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        if 'default_' + platform.system() in current_conf:
            return current_conf['default_' + platform.system()]
        return current_conf['default']

    @staticmethod
    def get(*key_hierarchy):
        """ Returns the current value of a given configuration entry.
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration entry value.
        """
        # Environment variable override
        # NOTE: will only work if a specific key is accessed!
        envvar = 'DACE_' + '_'.join(key_hierarchy)
        if envvar in os.environ:
            return os.environ[envvar]

        # Traverse the key hierarchy
        current_conf = Config._config
        for key in key_hierarchy:
            current_conf = current_conf[key]

        return current_conf

    @staticmethod
    def get_bool(*key_hierarchy):
        """ Returns the current value of a given boolean configuration entry.
            This specialization allows more string types to be converted to
            boolean, e.g., due to environment variable overrides.
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration entry value (as a boolean).
        """
        res = Config.get(*key_hierarchy)
        if isinstance(res, bool):
            return res
        return _env2bool(str(res))

    @staticmethod
    def append(*key_hierarchy, value=None, autosave=False):
        """ Appends to the current value of a given configuration entry
            and sets it. Example usage:
            `Config.append('compiler', 'cpu', 'args', value='-fPIC')`
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :param value: The value to append.
            :param autosave: If True, saves the configuration to the file
                             after modification.
            :return: Current configuration entry value.
        """
        # Traverse the key hierarchy up until the next to last element
        current_conf = Config._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] += value
        if autosave:
            Config.save()

        return current_conf[key_hierarchy[-1]]

    @staticmethod
    def set(*key_hierarchy, value=None, autosave=False):
        """ Sets the current value of a given configuration entry.
            Example usage:
            `Config.set('profiling', value=True)`
            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :param value: The value to set.
            :param autosave: If True, saves the configuration to the file
                             after modification.
        """
        # Traverse the key hierarchy up until the next to last element
        current_conf = Config._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] = value
        if autosave:
            Config.save()


# Code that runs when the module is loaded
Config.initialize()
