# Contributing to EasyScience

Thank you for your interest in contributing to **EasyScience**!

This guide explains how you can:

- Report issues
- Contribute code
- Improve documentation
- Suggest enhancements
- Interact with the EasyScience community

Whether you are an experienced developer or contributing for the first
time, this document walks you through the entire process step by step.

Please make sure you follow the EasyScience organization-wide
[Code of Conduct](https://github.com/easyscience/.github/blob/master/CODE_OF_CONDUCT.md).

---

## Table of Contents

- [How to Interact With This Project](#how-to-interact-with-this-project)
- [1. Understanding the Development Model](#1-understanding-the-development-model)
- [2. Getting the Code](#2-getting-the-code)
- [3. Setting Up the Development Environment](#3-setting-up-the-development-environment)
- [4. Creating a Branch](#4-creating-a-branch)
- [5. Implementing Your Changes](#5-implementing-your-changes)
- [6. Code Quality Checks](#6-code-quality-checks)
- [7. Opening a Pull Request](#7-opening-a-pull-request)
- [8. Continuous Integration (CI)](#8-continuous-integration-ci)
- [9. Code Review](#9-code-review)
- [10. Documentation Contributions](#10-documentation-contributions)
- [11. Reporting Issues](#11-reporting-issues)
- [12. Security Issues](#12-security-issues)
- [13. Releases](#13-releases)

---

## How to Interact With This Project

If you are not planning to contribute code, you may want to:

- 🐞 Report a bug — see [Reporting Issues](#11-reporting-issues)
- 🛡 Report a security issue — see
  [Security Issues](#12-security-issues)
- 💬 Ask a question or start a discussion at
  [Project Discussions](https://github.com/easyscience/core/discussions)

If you plan to contribute code or documentation, continue below.

---

## 1. Understanding the Development Model

Before you start coding, it is important to understand how development
works in this project.

### Branching Strategy

We use the following branches:

- `master` — stable releases only
- `develop` — active development branch
- Short-lived branches — feature or fix branches created for a single
  contribution and deleted after merge

> [!IMPORTANT]
>
> All normal contributions must target the `develop` branch.
>
> - Do **not** open Pull Requests against `master`
> - Always create your branch from `develop`
> - Always target `develop` when opening a Pull Request

See ADR easyscience/.github#12 for more details on the branching
strategy.

---

## 2. Getting the Code

### 2.1. If You Are an External Contributor

If you are not a core maintainer of this repository, follow these steps.

1. Open the repository page: `https://github.com/easyscience/core`

2. Click the **Fork** button (top-right corner). This creates your own
   copy of the repository.

3. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/core.git
   cd core
   ```

4. Add the original repository as `upstream`:

   ```bash
   git remote add upstream https://github.com/easyscience/core.git
   ```

5. Switch to the `develop` branch and update it:

   ```bash
   git fetch upstream
   git checkout develop
   git pull upstream develop
   ```

If you have contributed before, make sure your local `develop` branch is
up to date before starting new work. You can update it with:

```bash
git fetch upstream
git pull upstream develop
```

This ensures you are working on the latest version of the project.

### 2.2. If You Are a Core Team Member

Core team members can create branches directly in this repository and
therefore do not need to fork it, but the rest of the workflow remains
the same.

---

## 3. Setting Up the Development Environment

You need:

- Git
- Pixi

EasyScience projects use **Pixi** to manage the development environment.

To install Pixi, follow the official instructions:
https://pixi.prefix.dev/latest/installation/

You do **not** need to manually install Python. Pixi automatically:

- Creates the correct Python environment
- Installs all required dependencies
- Installs development tools (linters, formatters, test tools)

Set up the environment:

```bash
pixi install
pixi run post-install  # Install additional tooling
```

After this step, your development environment is ready.

See ADR easyscience/.github#63 for more details about using Pixi for
development.

---

## 4. Creating a Branch

Never work directly on `develop`.

Create a new branch:

```bash
git checkout -b my-change develop
```

> [!IMPORTANT]
>
> Use a clear and descriptive name, for example:
>
> - `improve-solver-speed`
> - `fix-boundary-condition`
> - `add-tutorial-example`

Clear branch names make reviews and history easier to understand.

---

## 5. Implementing Your Changes

While developing, make small, logical commits with clear messages.

Example:

```bash
git add .
git commit -m "Improve performance of time integrator for large systems"
```

---

## 6. Code Quality Checks

> [!IMPORTANT]
>
> When adding new functionality or making changes, make sure to add or
> update the following as needed:
>
> - 📘 docstrings
> - 🧪 unit tests

Before opening a Pull Request, always run:

```bash
pixi run check
```

This command:

- Validates the pyproject.toml file
- Checks for licence headers in code files
- Identifies linting and formatting issues in Python code
- Checks docstring linting and formatting issues in Python code
- Detects formatting issues in non-Python files (MD, YAML, TOML etc.)
- Checks linting issues in Jupyter notebooks (if applicable)
- Runs unit tests

A successful run should look like this:

```bash
pixi run pyproject-check.......................Passed
pixi run license-check.........................Passed
pixi run py-lint-check.........................Passed
pixi run py-format-check.......................Passed
pixi run docstring-lint-check..................Passed
pixi run nonpy-format-check....................Passed
pixi run notebook-lint-check...................Passed
pixi run unit-tests............................Passed
```

If something fails, read the error message carefully and fix the issue.

You can run individual checks, for example, to run only unit tests:

```bash
pixi run unit-tests
```

or to run only Python linting checks:

```bash
pixi run py-lint-check
```

Some formatting issues can be fixed automatically:

```bash
pixi run fix
```

If everything is correctly formatted, you will see:

```text
✅ All auto-formatting steps completed successfully!
```

This indicates that the auto-formatting pipeline completed successfully.
If you do not see this message and no error messages appear, try running
the command again.

If errors are reported, resolve them and re-run:

```bash
pixi run check
```

> [!IMPORTANT]
>
> All checks must pass before your Pull Request can be merged.

If you are unsure how to fix an issue, ask for help in your Pull Request
discussion.

---

## 7. Opening a Pull Request

Push your branch:

```bash
git push origin my-change
```

On GitHub:

- Click **Compare & Pull Request**
- Ensure the base branch is `develop`
- Write a clear and concise title
- Add a description explaining what changed and why
- Add the required `[scope]` label

### Pull Request Title

> [!IMPORTANT]
>
> The PR title appears in release notes and changelogs. It should be
> concise and informative.

Good examples:

- Improve performance of time integrator for large systems
- Fix incorrect boundary condition handling in solver
- Add adaptive step-size control to ODE solver
- Add tutorial for custom model configuration
- Refactor solver API for improved readability

### Required `[scope]` Label

> [!IMPORTANT]
>
> Each Pull Request must include a `[scope]` label, which is used for
> automatically suggesting version bumps when preparing a new release.

The available scopes are:

| Label                   | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `[scope] bug`           | Bug report or fix (major.minor.**PATCH**)                               |
| `[scope] documentation` | Documentation-only changes (major.minor.patch.**POST**)                 |
| `[scope] enhancement`   | Adds or improves features (major.**MINOR**.patch)                       |
| `[scope] maintenance`   | Code/tooling cleanup without feature or bug fix (major.minor.**PATCH**) |
| `[scope] significant`   | Breaking or major changes (**MAJOR**.minor.patch)                       |

See ADR easyscience/.github#33 for more details on the standardized
labeling scheme.

---

## 8. Continuous Integration (CI)

After opening a Pull Request:

- Automated checks run automatically
- You will see green checkmarks or red crosses

If checks fail:

1. Open the failing check
2. Read the logs
3. Fix the issue locally
4. Run `pixi run check`
5. Push your changes

The Pull Request updates automatically.

---

## 9. Code Review

All Pull Requests are reviewed by at least one core team member.

Code review is collaborative and aims to improve quality.

Do not take comments personally — they are meant to help.

To update your PR:

```bash
git add .
git commit -m "Address review comments"
git push
```

---

## 10. Documentation Contributions

> [!IMPORTANT]
>
> If your change affects user-facing functionality, update the project
> documentation accordingly — specifically the `nav:` (navigation)
> structure in `mkdocs.yml` and the relevant documentation Markdown
> files in `docs/docs/`.
>
> ```text
> 📁 docs
> ├── 📁 docs        - Markdown files for documentation
> │   └── ...
> └── 📄 mkdocs.yml  - Configuration file (navigation, theme, etc.)
> ```

This may include:

- API documentation
- Examples
- Tutorials
- Jupyter notebooks

Preview documentation locally:

```bash
pixi run docs-serve
```

Open the URL shown in the terminal to review your changes.

---

## 11. Reporting Issues

If you find a bug but cannot work on a fix, please consider opening an
issue.

When reporting an issue, it helps to:

- Search existing issues first.
- Provide clear reproduction steps.
- Include logs, screenshots, and environment details.

Clear and detailed reports help maintainers investigate and resolve
issues more effectively.

---

## 12. Security Issues

> [!IMPORTANT]
>
> Please do **not** report security vulnerabilities publicly.

If you discover a potential vulnerability, please contact the
maintainers privately so the issue can be investigated and addressed
responsibly.

---

## 13. Releases

Once your contribution is merged into `develop`, it will eventually be
included in the next stable release.

When enough changes have accumulated in `develop`, core team members
merge `develop` into `master` to prepare a new release. The release is
then tagged and published on GitHub and PyPI.

---

Thank you for contributing to EasyScience and the EasyScience ecosystem!
