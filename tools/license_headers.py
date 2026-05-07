# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Add, remove, or check SPDX headers in Python files."""

from __future__ import annotations

import argparse
import fnmatch
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

from git import Repo
from spdx_headers.core import find_repository_root
from spdx_headers.core import get_copyright_info
from spdx_headers.core import has_spdx_header
from spdx_headers.data import load_license_data
from spdx_headers.operations import add_header_to_single_file
from spdx_headers.operations import remove_header_from_single_file

LICENSE_DATABASE = load_license_data()


def load_pyproject(repo_path: Union[str, Path]) -> dict[str, Any]:
    """
    Load and return parsed ``pyproject.toml`` data for the repository.
    """
    repo_root = find_repository_root(repo_path)
    pyproject_path = repo_root / 'pyproject.toml'

    with pyproject_path.open('rb') as file_handle:
        return tomllib.load(file_handle)


def get_pyproject_value(pyproject_data: dict[str, Any], dotted_key: str) -> Any:
    """Return a nested ``pyproject.toml`` value from a dotted key."""
    value: Any = pyproject_data
    for part in dotted_key.split('.'):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def normalize_pattern(pattern: str) -> str:
    """Normalize an exclude pattern to a POSIX-style relative path."""
    normalized = Path(pattern).as_posix()
    if normalized.startswith('./'):
        normalized = normalized[2:]
    return normalized.rstrip('/')


def get_exclude_patterns(
    repo_path: Union[str, Path],
    exclude_values: list[str],
    exclude_from_pyproject_toml: Optional[str],
) -> list[str]:
    """
    Return normalized exclude patterns from CLI and ``pyproject.toml``.
    """
    pyproject_data = load_pyproject(repo_path)
    patterns: list[str] = []

    if exclude_from_pyproject_toml:
        value = get_pyproject_value(pyproject_data, exclude_from_pyproject_toml)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(
                f'{exclude_from_pyproject_toml} in pyproject.toml must be a list of strings.',
            )
        patterns.extend(value)

    for item in exclude_values:
        try:
            value = get_pyproject_value(pyproject_data, item)
        except KeyError:
            patterns.append(item)
            continue

        if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
            raise ValueError(f'{item} in pyproject.toml must be a list of strings.')
        patterns.extend(value)

    normalized_patterns: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        normalized = normalize_pattern(pattern)
        if normalized and normalized not in seen:
            normalized_patterns.append(normalized)
            seen.add(normalized)

    return normalized_patterns


def get_file_creation_year(file_path: Union[str, Path]) -> str:
    """Return the year the file was first added to Git history.

    If the year cannot be determined, fall back to the current year.
    """
    file_path = Path(file_path)

    repo = Repo(file_path, search_parent_directories=True)
    root = Path(repo.working_tree_dir).resolve()
    rel_path = file_path.resolve().relative_to(root)

    rel_path_git = rel_path.as_posix()

    log_output = repo.git.log(
        '--follow',
        '--diff-filter=A',
        '--reverse',
        '--format=%ad',
        '--date=format:%Y',
        '--',
        rel_path_git,
    ).strip()

    year = log_output.splitlines()[0].strip() if log_output else ''

    return year or str(datetime.now().year)


def get_org_url(repo_path: Union[str, Path]) -> str:
    """
    Return the organization URL derived from the repository source URL.
    """
    pyproject_data = load_pyproject(repo_path)
    repo_url = pyproject_data['project']['urls']['Source Code']
    return repo_url.rsplit('/', 1)[0]


def get_project_license(repo_path: Union[str, Path]) -> str:
    """Return the project license value from ``pyproject.toml``."""
    pyproject_data = load_pyproject(repo_path)
    return pyproject_data['project']['license']


def get_copyright_holder(repo_path: Union[str, Path]) -> str:
    """Return the repository copyright holder name."""
    _, name, _ = get_copyright_info(repo_path)
    return name


def add_spdx_header(
    target_file: Union[str, Path],
    *,
    license_key: str,
    copyright_holder: str,
    org_url: str,
) -> None:
    """Add SPDX headers to one file."""
    year = get_file_creation_year(target_file)

    add_header_to_single_file(
        filepath=target_file,
        license_key=license_key,
        license_data=LICENSE_DATABASE,
        year=year,
        name=copyright_holder,
        email=org_url,
    )


def is_excluded(relative_path: str, exclude_patterns: list[str]) -> bool:
    """Return whether a relative path should be excluded."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(relative_path, pattern):
            return True
        if relative_path == pattern:
            return True
        if relative_path.startswith(f'{pattern}/'):
            return True
    return False


def iter_python_files(
    paths: list[str],
    *,
    repo_root: Path,
    exclude_patterns: list[str],
    parser: argparse.ArgumentParser,
) -> list[Path]:
    """Collect Python files under the given paths after exclusions."""
    files: list[Path] = []
    seen: set[Path] = set()

    for base_dir in paths:
        base_path = Path(base_dir)
        if not base_path.exists():
            parser.error(f'Path does not exist: {base_dir}')

        if base_path.is_file():
            candidates = [base_path] if base_path.suffix == '.py' else []
        else:
            candidates = sorted(base_path.rglob('*.py'))

        for py_file in candidates:
            resolved = py_file.resolve()
            try:
                relative_path = resolved.relative_to(repo_root).as_posix()
            except ValueError:
                relative_path = py_file.as_posix()

            if is_excluded(relative_path, exclude_patterns):
                continue

            if resolved not in seen:
                files.append(py_file)
                seen.add(resolved)

    return files


def run_add(
    files: list[Path],
    *,
    license_key: str,
    copyright_holder: str,
    org_url: str,
) -> int:
    """Add SPDX headers to all selected files."""
    for py_file in files:
        add_spdx_header(
            py_file,
            license_key=license_key,
            copyright_holder=copyright_holder,
            org_url=org_url,
        )
    return 0


def run_remove(files: list[Path]) -> int:
    """Remove SPDX headers from all selected files."""
    for py_file in files:
        remove_header_from_single_file(py_file)
    return 0


def run_check(files: list[Path]) -> int:
    """Check SPDX headers in all selected files."""
    missing_files = [py_file for py_file in files if not has_spdx_header(py_file)]

    if not missing_files:
        print('✓ All Python files have valid SPDX headers.')
        return 0

    print('✗ The following files are missing SPDX headers:')
    for py_file in missing_files:
        print(f'  - {py_file.as_posix()}')
    print(f'\nFound {len(missing_files)} files without SPDX headers.')
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Add, remove, or check SPDX headers in Python files.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    for command_name in ('check', 'remove', 'add'):
        command_parser = subparsers.add_parser(command_name)
        command_parser.add_argument(
            'paths',
            nargs='+',
            help='Relative paths to scan (e.g. src tests)',
        )
        command_parser.add_argument(
            '--exclude',
            nargs='*',
            default=[],
            help='Exclude paths, glob patterns, or pyproject dotted keys.',
        )
        command_parser.add_argument(
            '--exclude-from-pyproject-toml',
            help='Read exclude patterns from a dotted key in pyproject.toml.',
        )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the SPDX header CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_path = Path('.').resolve()
    repo_root = find_repository_root(repo_path).resolve()
    exclude_patterns = get_exclude_patterns(
        repo_path,
        args.exclude,
        args.exclude_from_pyproject_toml,
    )
    files = iter_python_files(
        args.paths,
        repo_root=repo_root,
        exclude_patterns=exclude_patterns,
        parser=parser,
    )

    if args.command == 'check':
        return run_check(files)

    if args.command == 'remove':
        return run_remove(files)

    license_key = get_project_license(repo_path)
    copyright_holder = get_copyright_holder(repo_path)
    org_url = get_org_url(repo_path)
    return run_add(
        files,
        license_key=license_key,
        copyright_holder=copyright_holder,
        org_url=org_url,
    )


if __name__ == '__main__':
    raise SystemExit(main())
