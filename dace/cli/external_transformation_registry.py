# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
DaCe Transformation Repository Manager

CLI tool for managing external transformation repositories.
Allows users to register, clone, and manage transformation packages.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
import dace
import shutil


class TransformationRepoManager:

    def _set_path(self):
        """
        Sets up the paths for external transformation repositories and registry files.
        Ensures the external directory and necessary __init__.py files exist.
        """
        # Get default path for external transformations, eval $HOME if present
        base_path_home_unevaluated_str = dace.config.Config.get("external_transformations_path")
        home_str = str(Path.home())
        base_path_str = base_path_home_unevaluated_str.replace('$HOME', home_str)
        base_path = Path(base_path_str)
        self.module_root_path = Path(dace.__file__).resolve().parent.parent
        if not base_path.is_absolute():
            base_path = self.module_root_path / base_path
        self.base_path = base_path
        self.external_path = self.base_path
        self.registry_file = self.base_path / "local_external_transformation_repositories.json"

        # Ensure external directory exists
        self.external_path.mkdir(parents=True, exist_ok=True)

        # Initialize empty __init__.py if it doesn't exist
        init_file = self.external_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        base_init_file = self.external_path / ".." / "__init__.py"
        if not base_init_file.exists():
            base_init_file.touch()
        if not self.registry_file.exists():
            with open(self.registry_file, 'w') as f:
                json.dump({}, f, indent=2)

    def __init__(self):
        """
        Initializes the TransformationRepoManager by setting up paths.
        """
        self._set_path()  # In a function as we might need to re-read it, if someone changed the config

    def _load_registry(self, filepath: str) -> Dict:
        """
        Loads the repository registry from a JSON file.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            Dict: The loaded registry as a dictionary.
        """
        self._set_path()
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load registry file: {e}")
            return {}

    def _save_registry(self, registry: Dict) -> None:
        """
        Saves the repository registry to a JSON file.

        Args:
            registry (Dict): The registry dictionary to save.
        """
        self._set_path()
        reg_file = self.registry_file
        try:
            with open(reg_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            print(f"Error: Could not save registry file: {e}")
            sys.exit(1)

    def _get_repo_name(self, url: str) -> str:
        """
        Extracts the repository name from a given URL.

        Args:
            url (str): The URL of the repository.

        Returns:
            str: The extracted repository name.

        Raises:
            ValueError: If the repository name cannot be extracted.
        """
        self._set_path()
        parsed = urlparse(url)
        if parsed.path:
            path = parsed.path.strip('/')
            if path.endswith('.git'):
                path = path[:-4]
            return os.path.basename(path)

        # Handle SSH format
        if '@' in url and ':' in url:
            path = url.split(':')[-1]
            if path.endswith('.git'):
                path = path[:-4]
            return os.path.basename(path)

        raise ValueError(f"Could not extract repository name from URL: {url}")

    def _clone_repository(self, url: str, target_path: Path) -> bool:
        """
        Clones a git repository to the specified target path.

        Args:
            url (str): The URL of the repository to clone.
            target_path (Path): The path where the repository should be cloned.

        Returns:
            bool: True if cloning was successful, False otherwise.
        """
        self._set_path()
        try:
            # Remove existing directory if it exists
            if target_path.exists():
                import shutil
                shutil.rmtree(target_path)

            # Clone the repository
            result = subprocess.run(['git', 'clone', url, str(target_path)],
                                    capture_output=True,
                                    text=True,
                                    check=False)

            if result.returncode != 0:
                print(f"Error cloning repository: {result.stderr}")
                return False

            return True

        except FileNotFoundError:
            print("Error: git command not found. Please install git.")
            return False
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return False

    def add_repository(self, url: str, name: Optional[str] = None, force=False) -> bool:
        """
        Clones a repository and adds it as a new transformation repository.

        Args:
            url (str): The URL of the repository to add.
            name(Optional[str]): Optional name for the repository.
            force (bool): If True, overwrites an existing repository with the same path.

        Returns:
            bool: True if the repository was added successfully, False otherwise.
        """
        self._set_path()
        # Get repository name
        if name is None:
            try:
                name = self._get_repo_name(url)
            except ValueError as e:
                print(f"Error: {e}")
                return False

        # Load existing registry
        registry = self._load_registry(filepath=self.registry_file)

        # Check if repository already exists
        if name in registry and not force:
            print(f"Repository '{name}' already exists, use --force.")
            return False

        # Clone repository
        target_path = self.external_path / name
        print(f"Cloning repository '{name}' to {target_path}...")

        if not self._clone_repository(url, target_path):
            return False

        # Add to registry
        registry[name] = {
            'url': url,
        }

        self._save_registry(registry)
        print(f"Successfully added repository '{name}'")
        return True

    def remove_repository(self, name: str) -> bool:
        """
        Removes a transformation repository by name. (Does not delete the cloned repository or the directory)

        Args:
            name (str): The name of the repository to remove.

        Returns:
            bool: True if the repository was removed successfully, False otherwise.
        """
        self._set_path()
        registry = self._load_registry(filepath=self.registry_file)

        if name not in registry:
            print(f"Error: Repository '{name}' not found")
            return False

        # Remove directory
        target_path = self.external_path / name
        if target_path.exists():
            import shutil
            shutil.rmtree(target_path)
            print(f"Removed directory: {target_path}")

        # Remove from registry
        del registry[name]
        self._save_registry(registry)
        print(f"Successfully removed repository '{name}'")
        return True

    def list_repositories(self) -> None:
        """
        Lists all registered transformation repositories.
        """
        self._set_path()
        print(self.registry_file)
        registry = self._load_registry(filepath=self.registry_file)

        if not registry:
            print("No repositories registered.")
            return

        print("Registered transformation repositories:")
        print("-" * 50)
        for name, info in registry.items():
            status = "✓" if (self.external_path / name).exists() else "✗"
            print(f"{status} {name}")
            print(f"  URL: {info['url']}")
            print(f"  Path: {info['path']}")
            print()

    def load_all_repositories(self, filepath: str, force: bool) -> bool:
        """
        Loads (clones) all repositories from the registry if they are missing.

        Args:
            filepath (str): Loads from the registry JSON file specified by the path.
            Format is a list involving repository names as keys and the repository URL as value. An example:
                {
                    <repo_name>: {
                        "url": <repo_url>,
                    },
                    <...>
                }
        Returns:
            bool: True if all repositories were loaded successfully, False otherwise.
        """
        self._set_path()
        registry = self._load_registry(filepath=filepath)

        if Path(filepath) == self.registry_file:
            print(
                f"Warning: Can't load from registry file {self.registry_file} used to track registered repositories. Use a different file to load repositories."
            )
            return False

        if not registry:
            print("No repositories in registry to load.")
            return True

        success = True
        for name, info in registry.items():
            self.add_repository(url=info['url'], name=name, force=force)

        if success:
            print("All repositories loaded successfully.")
        else:
            print("Some repositories failed to load.")

        return success


def main():
    """
    Entry point for the CLI tool. Parses command-line arguments and executes the corresponding command.
    """
    parser = argparse.ArgumentParser(description="Manage DaCe transformation repositories",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  %(prog)s add https://github.com/user/my-transformations.git
  %(prog)s add https://github.com/user/repo.git --name custom-name
  %(prog)s list
  %(prog)s remove my-transformations
  %(prog)s load-from-file my_repos.json
        format:
            {
                <repo_name>: {
                    "url": <repo_url>,
                    "path": <path_relative_to_base_path_in_config_schema>
                },
                <...>
            }
        """)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add and clone a transformation repository')
    add_parser.add_argument('url', help='Git URL of the repository')
    add_parser.add_argument('--name', help='Custom name for the repository')
    add_parser.add_argument('--force', action='store_true', help='Overwrite existing repository')

    remove_parser = subparsers.add_parser('remove', help='Remove a transformation repository')
    remove_parser.add_argument('name', help='Name of the repository to remove')

    list_parser = subparsers.add_parser('list', help='List all registered repositories')

    load_from_json = subparsers.add_parser('load-from-file', help='Load repositories from json file')
    load_from_json.add_argument('filepath', help='Filepath of the JSON file containing repositories')
    load_from_json.add_argument('--force', action='store_true', help='Overwrite existing repositories')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Initialize manager
    manager = TransformationRepoManager()

    # Execute command
    if args.command == 'add':
        success = manager.add_repository(args.url, args.name, args.force)
        sys.exit(0 if success else 1)

    elif args.command == 'remove':
        success = manager.remove_repository(args.name)
        sys.exit(0 if success else 1)

    elif args.command == 'list':
        manager.list_repositories()

    elif args.command == 'load-from-file':
        success = manager.load_all_repositories(args.filepath, args.force)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
