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

class TransformationRepoManager:
    def _set_path(self):
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
        self.registry_file = self.module_root_path / "external_transformation_repositories.json"
        self.local_registry_file = self.base_path / "local_external_transformation_repositories.json"

        # Ensure external directory exists
        self.external_path.mkdir(parents=True, exist_ok=True)

        # Initialize empty __init__.py if it doesn't exist
        init_file = self.external_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        base_init_file = self.external_path / ".." / "__init__.py"
        if not base_init_file.exists():
            base_init_file.touch()

    def __init__(self):
        self._set_path()  # In a function as we might need to re-read it, if someone changed the config

    def _load_registry(self, local: bool) -> Dict:
        self._set_path()
        reg_file = self.local_registry_file if local else self.registry_file
        if not reg_file.exists():
            return {}

        try:
            with open(reg_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load registry file: {e}")
            return {}

    def _save_registry(self, registry: Dict, local: bool) -> None:
        self._set_path()
        reg_file = self.local_registry_file if local else self.registry_file
        try:
            with open(reg_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            print(f"Error: Could not save registry file: {e}")
            sys.exit(1)

    def _get_repo_name(self, url: str) -> str:
        self._set_path()
        """Extract repository name from URL."""
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
        self._set_path()
        """Clone a git repository to the target path."""
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
        self._set_path()
        # Get repository name
        if name is None:
            try:
                name = self._get_repo_name(url)
            except ValueError as e:
                print(f"Error: {e}")
                return False

        # Load existing registry
        registry = self._load_registry(local=True)

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
            'path': str(target_path.relative_to(self.base_path)),
            'cloned_at': str(target_path.relative_to(self.base_path))
        }

        self._save_registry(registry, local=True)
        print(f"Successfully added repository '{name}'")
        return True

    def remove_repository(self, name: str) -> bool:
        self._set_path()
        registry = self._load_registry(local=True)

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
        self._save_registry(registry, local=True)
        print(f"Successfully removed repository '{name}'")
        return True

    def list_repositories(self) -> None:
        self._set_path()
        registry = self._load_registry(local=True)

        if not registry:
            print("No repositories registered.")
            return

        print("Locally registered transformation repositories:")
        print("-" * 50)
        for name, info in registry.items():
            status = "✓" if (self.external_path / name).exists() else "✗"
            print(f"{status} {name}")
            print(f"  URL: {info['url']}")
            print(f"  Path: {info['path']}")
            print()

    def load_all_repositories(self, local: bool) -> bool:
        self._set_path()
        """Load all repositories from the registry (clone missing ones).

        Returns:
            bool: True if all successful, False if any failed
        """
        registry = self._load_registry(local=local)

        if not registry:
            print("No repositories in registry to load.")
            return True

        success = True
        for name, info in registry.items():
            target_path = self.external_path / name

            if target_path.exists():
                print(f"Repository '{name}' already exists, skipping...")
                continue

            print(f"Cloning missing repository '{name}'...")
            if not self._clone_repository(info['url'], target_path):
                success = False
                continue

        if success:
            print("All repositories loaded successfully.")
        else:
            print("Some repositories failed to load.")

        return success


def main():
    parser = argparse.ArgumentParser(description="Manage DaCe transformation repositories",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  %(prog)s add https://github.com/user/my-transformations.git
  %(prog)s add https://github.com/user/repo.git --name custom-name
  %(prog)s list
  %(prog)s remove my-transformations
  %(prog)s load-all-local
  %(prog)s load-all-suggested
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

    load_all_parser_local = subparsers.add_parser('load-all-local',
                                                  help='Load all locally added repositories from registry')

    load_all_parser_suggest = subparsers.add_parser('load-all-suggested',
                                                    help='Load all suggested repositories from registry')

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

    elif args.command == 'load-all-local':
        success = manager.load_all_repositories(local=True)
        sys.exit(0 if success else 1)

    elif args.command == 'load-all-suggested':
        success = manager.load_all_repositories(local=False)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
