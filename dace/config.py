# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import os
import platform
import tempfile
import threading
import io
from typing import Any, Dict, Optional
import yaml
import warnings


@contextlib.contextmanager
def set_temporary(*path, value):
    """
    Temporarily set configuration value at ``path`` to value, and reset it after the context manager exits.

    Example::

        print(Config.get("compiler", "build_type")
        with set_temporary("compiler", "build_type", value="Debug"):
            print(Config.get("compiler", "build_type")
        print(Config.get("compiler", "build_type")
    """
    if len(path) == 1 and '.' in path[0]:
        path = tuple(path[0].split('.'))
    old_value = Config.get(*path)
    Config.set(*path, value=value)
    try:
        yield Config
    finally:
        Config.set(*path, value=old_value)


@contextlib.contextmanager
def temporary_config():
    """
    Creates a context where all configuration options changed will be reset when the context exits.

    Example::

        with temporary_config():
            Config.set("testing", "serialization", value=True)
            Config.set("optimizer", "autooptimize", value=True)
            foo()
    """
    with tempfile.TemporaryFile(mode='w+t') as fp:
        Config.save(file=fp)
        try:
            yield Config
        finally:
            fp.seek(0)  # rewind to the beginning of the file.
            Config.load(file=fp)


def _env2bool(envval):
    """
    Converts an arbitrary value to boolean.

    :param envval: Arbitrary value.
    :return: True if the input value matches a valid TRUE
             value, or False otherwise.
    """
    return str(envval).lower() in ['true', '1', 'y', 'yes', 'on', 'verbose']


class _ConfigData(threading.local):
    """Thread local data storage for the configuration scheme.

    Note that DaCe on its own is not thread safe, however, the tests
    impose this requirement.
    """

    def __init__(self) -> None:
        self.default_filename = '.dace.conf'
        self._config = {}
        self._config_metadata = {}
        self._cfg_filename = None
        self._default_cfg_path = None
        self._metadata_filename = None
        self._initialize()

    def cfg_filename(self):
        return self._cfg_filename

    @staticmethod
    def env_name_for(*key_hierarchy: str) -> str:
        """
        Returns the environment variable name of a configuration key, e.g.
        ``DACE_compiler_build_type`` for ``('compiler', 'build_type')``.

        :param key_hierarchy: The key path of the configuration entry.
        :return: The environment variable name.
        """
        return '_'.join(('DACE', ) + key_hierarchy)

    @staticmethod
    def coerce_env_value(envval: str, metadata: Dict[str, Any], envvar: str):
        """
        Coerces an environment variable string to the schema-declared type of a
        configuration entry.

        Coercion exists for compatibility with previous versions, where
        ``_config`` values carried the types the YAML loader produced for
        schema defaults and configuration-file entries (``false`` loads as
        ``bool``, ``5`` as ``int``): an environment-derived value is made to
        match what the YAML loader would produce for the same text, so
        ``get()`` returns the same type regardless of where a value came
        from. Note that the schema-declared type is not enforced anywhere —
        a hand-edited file may store any YAML type for any key, unvalidated
        (that laxity predates this function and is unchanged by it).

        :param envval: The raw environment variable value.
        :param metadata: The schema metadata of the configuration entry.
        :param envvar: The environment variable name (for diagnostics).
        :return: The coerced value.
        :raise ValueError: If the value cannot be coerced to the declared type.
        """
        entry_type = metadata['type']
        if entry_type == 'bool':
            return _env2bool(envval)
        if entry_type == 'int':
            return int(envval)
        if entry_type == 'float':
            return float(envval)
        if entry_type == 'list':
            result = yaml.load(envval, Loader=yaml.SafeLoader)
            if not isinstance(result, list):
                raise ValueError(f'{envvar} does not contain a list: {envval!r}')
            return result
        # Strings (and 'any'-typed entries) are kept verbatim
        return envval

    def add_defaults(self, config, metadata):
        """
        Adds defaults to the configuration from metadata.

        Fills only the keys missing from ``config`` with their schema
        defaults; the environment is handled separately by
        :func:`apply_env` when the configuration is loaded.

        :param config: The (sub-)configuration dictionary to fill.
        :param metadata: The schema metadata of ``config``.
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
                modified |= self.add_defaults(config[k], v['required'])
                continue
            # Key already exists in configuration, nothing to add
            if k in config:
                continue
            modified = True
            # Empty list initialization (if no default is specified)
            if v['type'] == 'list' and 'default' not in v:
                config[k] = []
            # Per-OS default
            elif 'default_' + osname in v:
                config[k] = v['default_' + osname]
            else:
                config[k] = v['default']
        return modified

    def apply_env(self, config, metadata, key_path=()):
        """
        Applies ``DACE_*`` environment variables onto the configuration.

        Runs when the configuration is loaded, after the configuration
        file and the schema defaults were filled in, and overwrites the
        affected entries: the source precedence at load time is, with
        increasing priority, the schema default, the configuration file
        (``.dace.conf``, or the file named by ``DACE_CONFIG``), and the
        environment. Values set explicitly afterwards through
        :func:`Config.set` (including :func:`set_temporary` /
        :func:`temporary_config`) have the highest priority, since the
        environment is never consulted again until the next load.
        Environment values are coerced to the schema-declared type (see
        :func:`coerce_env_value`); a value that cannot be coerced is
        reported with a warning and ignored.

        :param config: The (sub-)configuration dictionary to modify.
        :param metadata: The schema metadata of ``config``.
        :param key_path: The key path of ``config`` (empty at the root).
        """
        for k, v in metadata.items():
            if v['type'] == 'dict':
                self.apply_env(config.setdefault(k, {}), v['required'], key_path + (k, ))
                continue
            envvar = self.env_name_for(*key_path, k)
            if envvar in os.environ:
                try:
                    config[k] = self.coerce_env_value(os.environ[envvar], v, envvar)
                except (ValueError, yaml.YAMLError) as ex:
                    warnings.warn(f'Ignoring environment variable {envvar}: {ex}')

    def _initialize(self):
        """Initialize `self`, loads the specified configuration file.
        The function is automatically called by the constructor.
        """

        # If already initialized, skip
        if self._config_metadata:
            return

        # Override default configuration file path
        if 'DACE_CONFIG' in os.environ:
            default_cfg_filename = os.environ['DACE_CONFIG']
        else:
            home = os.path.expanduser("~")
            default_cfg_filename = os.path.join(home, self.default_filename)

        self._default_cfg_path = default_cfg_filename

        dace_path = os.path.dirname(os.path.abspath(__file__))
        self._metadata_filename = os.path.join(dace_path, 'config_schema.yml')

        # Load configuration schema (for validation and defaults)
        self.load_schema(filename=None)

        # Priority order: current working directory, default configuration file (DACE_CONFIG), then ~/.dace.conf
        for filename in [self.default_filename, default_cfg_filename]:
            self._cfg_filename = filename
            try:
                if os.path.isfile(filename):
                    self.load()
                    break
            except (FileNotFoundError, PermissionError, OSError):
                # If any filesystem-related error happened during file load, move on to next candidate
                continue
        else:
            # No configuration file exists, so load() above never ran; build the
            # configuration the same way load() would, minus the file: schema
            # defaults first, then the environment on top.
            self._cfg_filename = None
            self._config = {}
            self.add_defaults(self._config, self._config_metadata['required'])
            self.apply_env(self._config, self._config_metadata['required'])

        # Migration of very old-format configuration files: the legacy 'execution' entry marks a
        # `dace.conf` written by very old DaCe versions, which saved every configuration entry.
        # Such a file is rewritten once in the new format, which keeps only the nondefault entries.
        # Note that the environment has already been applied at this point, so `DACE_*` values
        # that are set while the migration runs are persisted into the file as well. Files already
        # in the new format are never written back.
        if 'execution' in self._config and self._cfg_filename:
            self.save(all=False)

    def load(self, filename: Optional[str] = None, file: Optional[io.FileIO] = None):
        if file is not None:
            assert filename is None
            self._config = yaml.load(file.read(), Loader=yaml.SafeLoader)
        else:
            with open(filename if filename else self._cfg_filename, 'r') as f:
                self._config = yaml.load(f.read(), Loader=yaml.SafeLoader)

        if self._config is None:
            self._config = {}

        # Add defaults from metadata, then apply the environment on top
        self.add_defaults(self._config, self._config_metadata['required'])
        self.apply_env(self._config, self._config_metadata['required'])

    def load_schema(self, filename: Optional[str] = None):
        if filename is None:
            filename = self._metadata_filename
        with open(filename, 'r') as f:
            self._config_metadata = yaml.load(f.read(), Loader=yaml.SafeLoader)

    def extend(self, schema_filename: str):
        """
        Extends the current configuration schema with another schema from file.

        :param schema_filename: The schema file to load.
        """
        with open(schema_filename, 'r') as f:
            new_metadata = yaml.load(f.read(), Loader=yaml.SafeLoader)

        def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dicts(d1[k], v)
                else:
                    d1[k] = v

        merge_dicts(self._config_metadata['required'], new_metadata['required'])
        self.add_defaults(self._config, new_metadata['required'])

    def save(self, path: Optional[str] = None, all: bool = False, file: Optional[io.FileIO] = None):
        what_to_save = self._config if all else self.nondefaults()
        if file is not None:
            assert path is None, "Specified both `path` and `file` in `Config.save()`."
            yaml.dump(what_to_save, file, default_flow_style=False)

        elif path is not None:
            assert file is None, "Specified both `path` and `file` in `Config.save()`."
            with open(path, 'w') as f:
                yaml.dump(what_to_save, f, default_flow_style=False)

        elif self._default_cfg_path is not None:
            # Neither a path nor a file was specified, but a default configuration file
            #  path is known. Use that.
            self.save(path=self._default_cfg_path, all=all, file=None)

        else:
            # Neither a path nor a file was specified of a configuration file is known.
            #  Try the default ones.
            for filename in [self._default_cfg_path, self.default_filename]:
                if not os.path.isfile(filename):
                    continue
                try:
                    self.save(path=filename, all=all)
                    self._cfg_filename = filename
                    return
                except (PermissionError, OSError):
                    # If any filesystem-related error happened during file save, move on to next candidate
                    continue
            warnings.warn('No DaCe configuration file was able to be saved')

    def get_metadata(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy
        current_conf = self._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        return current_conf

    def get_default(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy
        current_conf = self._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        if 'default_' + platform.system() in current_conf:
            return current_conf['default_' + platform.system()]
        return current_conf['default']

    def get(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy
        current_conf = self._config
        for key in key_hierarchy:
            current_conf = current_conf[key]

        return current_conf

    def get_bool(self, *key_hierarchy):
        res = self.get(*key_hierarchy)
        if isinstance(res, bool):
            return res
        # Values loaded from a configuration file may be strings
        return _env2bool(str(res))

    def append(self, *key_hierarchy, value):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy up until the next to last element
        current_conf = self._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] += value

        return current_conf[key_hierarchy[-1]]

    def set(self, *key_hierarchy, value):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy up until the next to last element
        current_conf = self._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] = value

    def nondefaults(self) -> Dict[str, Any]:
        current_conf = self._config
        defaults = self._config_metadata
        system_default_key = 'default_' + platform.system()

        def traverse(conf: Dict[str, Any], defaults: Dict[str, Any], result: Dict[str, Any]):
            for k, v in conf.items():
                if k not in defaults:  # Configuration entry no longer exists
                    continue
                elif 'required' in defaults[k]:  # Traverse further
                    internal = {}
                    traverse(v, defaults[k]['required'], internal)
                    if internal:
                        result[k] = internal
                elif system_default_key in defaults[k]:
                    if v != defaults[k][system_default_key]:
                        result[k] = v
                elif 'default' in defaults[k]:
                    if v != defaults[k]['default']:
                        result[k] = v

        output = {}
        traverse(current_conf, defaults['required'], output)
        return output


class Config(object):
    """Interface to the DaCe hierarchical configuration file.

    :note: The data is stored inside a thread local, aka. `threading.local`,
        variable. This means that in the beginning every thread is initialized
        with the _default_ setting.
    """

    _data = _ConfigData()

    @staticmethod
    def cfg_filename():
        """
        Returns the current configuration file path.
        """
        return Config._data.cfg_filename()

    @staticmethod
    def extend(schema_filename: str):
        """
        Extends the current configuration schema with another schema from file.

        :param schema_filename: The schema file to load.
        """
        return Config._data.extend(schema_filename=schema_filename)

    @staticmethod
    def load(filename: Optional[str] = None, file: Optional[io.FileIO] = None):
        """
        Loads a configuration from an existing file.

        :param filename: The file to load. If unspecified,
                         uses default configuration file.
        :param file: Load the configuration from the file object.
        """
        return Config._data.load(filename=filename, file=file)

    @staticmethod
    def load_schema(filename: Optional[str] = None):
        """
        Loads a configuration schema from an existing file.

        :param filename: The file to load. If unspecified,
                         uses default schema file.
        """
        return Config._data.load(filename=filename)

    @staticmethod
    def save(path: Optional[str] = None, all: bool = False, file: Optional[io.FileIO] = None):
        """
        Saves the current configuration to a file.

        :param path: The file to save to. If unspecified,
                     uses default configuration file.
        :param all: If False, only saves non-default configuration entries.
                    Otherwise saves all entries.
        :param file: A file object to use directly.
        """
        return Config._data.save(path=path, all=all, file=file)

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
        return Config._data.get_metadata(*key_hierarchy)

    @staticmethod
    def get_default(*key_hierarchy):
        """ Returns the default value of a given configuration entry.
            Takes into account current operating system.

            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Default configuration value.
        """
        return Config._data.get_default(*key_hierarchy)

    @staticmethod
    def get(*key_hierarchy):
        """
        Returns the current value of a given configuration entry.

        :param key_hierarchy: A tuple of strings leading to the
                                configuration entry.
                                For example: ('a', 'b', 'c') would be
                                configuration entry c which is in the
                                path a->b.
        :return: Configuration entry value.
        """
        return Config._data.get(*key_hierarchy)

    @staticmethod
    def get_bool(*key_hierarchy):
        """ Returns the current value of a given boolean configuration entry.
            This specialization allows more string types to be converted to
            boolean, e.g., when a configuration file stores the value as a
            string.

            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration entry value (as a boolean).
        """
        return Config._data.get_bool(*key_hierarchy)

    @staticmethod
    def append(*key_hierarchy, value=None):
        """
        Appends to the current value of a given configuration entry
        and sets it.

        Never writes the configuration file; use :func:`Config.save`
        explicitly to persist configuration changes.

        :param key_hierarchy: A tuple of strings leading to the
                                configuration entry.
                                For example: ('a', 'b', 'c') would be
                                configuration entry c which is in the
                                path a->b.
        :param value: The value to append.
        :return: Current configuration entry value.

        Examples::

            Config.append('compiler', 'cpu', 'args', value='-fno-plt')
        """
        return Config._data.append(*key_hierarchy, value=value)

    @staticmethod
    def set(*key_hierarchy, value=None):
        """
        Sets the current value of a given configuration entry.

        Never writes the configuration file; use :func:`Config.save`
        explicitly to persist configuration changes.

        :param key_hierarchy: A tuple of strings leading to the
                              configuration entry.
                              For example: ('a', 'b', 'c') would be
                              configuration entry c which is in the
                              path a->b.
        :param value: The value to set.

        Examples::

            Config.set('profiling', value=True)
        """
        return Config._data.set(*key_hierarchy, value=value)

    def nondefaults(self) -> Dict[str, Any]:
        return Config._data.nondefaults()
