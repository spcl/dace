# Dace Transformation Repository Manager

## Table of Contents

- [Quick Start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Quick Start

```bash
# 1. Add your first transformation repository
python transformation_manager.py add https://github.com/username/my-transformations.git

# 2. List all registered repositories
python transformation_manager.py list

# 3. Load all suggested transformation repositories (from experimental_transformation_repositories.json)
# Locally added repos are saved in local_experimental_transformation_repositories.json
python transformation_manager.py load-all-suggested
```

## Commands

### `add` - Add a Transformation Repository

Register and clone a new transformation repository.

**Syntax**:
```bash
python transformation_manager.py add <git-url> [--name <custom-name>] [--force]
```

**Parameters**:
- `<git-url>`: Git URL of the repository (HTTPS, SSH, or Git protocol)
- `--name <custom-name>`: Optional custom name for the repository
- `--force`: Overwrite existing repository if it exists

---

### `list` - List All Repositories

Display all registered transformation repositories and their status.

---

### `remove` - Remove a Repository

Remove a transformation repository from both filesystem and registry.

1. Removes the repository directory
2. Removes entry from `local_experimental_transformation_repositories.json`
---

### `load-all-suggested` - Load All Repositories

Clone all repositories listed in the suggested registry file that are not currently present.

### `load-all-suggested` - Load All Repositories

Clone all repositories listed in the local registry file (if created by the user, then this is nop operation, if copied from a file it will clone) that are not currently present.
