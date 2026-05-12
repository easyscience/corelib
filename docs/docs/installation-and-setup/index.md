---
icon: material/cog-box
---

# :material-cog-box: Installation & Setup

**EasyScience** is a cross-platform Python library compatible with
**Python 3.11** through **3.13**.

To install and set up EasyScience, we recommend using
[**Pixi**](https://pixi.prefix.dev), a modern package manager for
Windows, macOS, and Linux.

??? note "Main benefits of using Pixi"

    - **Ease of use**: Pixi simplifies the installation process, making it
      accessible even for users with limited experience in package management.
    - **Python version control**: Pixi allows specifying and managing different
      Python versions for each project, ensuring compatibility.
    - **Isolated environments**: Pixi creates isolated environments for each
      project, preventing conflicts between different package versions.
    - **PyPI and Conda support**: Pixi can install packages from both PyPI and
      Conda repositories, providing access to a wide range of libraries.

An alternative installation method using the traditional **pip** package
manager is also provided.

## Installing with Pixi <small>recommended</small> { #installing-with-pixi data-toc-label="Installing with Pixi" }

This section describes the simplest way to set up EasyScience using
**Pixi**.

#### Installing Pixi

- Install Pixi by following the instructions on the
  [official Pixi Installation Guide](https://pixi.prefix.dev/latest/installation).

#### Setting up EasyScience with Pixi

<!-- prettier-ignore-start -->

- Choose a project location (local drive recommended).

    ??? warning ":fontawesome-brands-windows: Windows + OneDrive"

        We **do not recommend creating a Pixi project inside OneDrive or other
        synced folders**.

        By default, Pixi creates the virtual environment inside the project
        directory (in `.pixi/`). On Windows, synced folders such as OneDrive
        may cause file‑system issues (e.g., path-length limitations or
        restricted link operations), which can lead to unexpected install
        errors or environments being recreated.

        Instead, create your project in a **local directory on your drive**
        where you have full write permissions.

<!-- prettier-ignore-end -->

- Initialize a new Pixi project and navigate into it:
  ```txt
  pixi init easyscience
  cd easyscience
  ```
- Set the Python version for the Pixi environment (e.g., 3.13):
  ```txt
  pixi add python=3.13
  ```
- Add EasyScience to the Pixi environment from PyPI:
  ```txt
  pixi add --pypi easyscience
  ```
- Add a Pixi task to run EasyScience commands easily:
  ```txt
  pixi task add easyscience "python -m easyscience"
  ```

#### Updating Pixi and EasyScience

- To update all packages in the Pixi environment, including EasyScience:
  ```txt
  pixi update
  ```
- To update Pixi itself to the latest version:
  ```txt
  pixi self-update
  ```

#### Uninstalling Pixi

- Follow the
  [official Pixi Guide](https://pixi.prefix.dev/latest/installation/#uninstall).

## Classical Installation

This section describes how to install EasyScience using the traditional
method with **pip**. It is assumed that you are familiar with Python
package management and virtual environments.

### Environment Setup <small>optional</small> { #environment-setup data-toc-label="Environment Setup" }

We recommend using a **virtual environment** to isolate dependencies and
avoid conflicts with system-wide packages. If any issues arise, you can
simply delete and recreate the environment.

#### Creating and Activating a Virtual Environment:

<!-- prettier-ignore-start -->

- Create a new virtual environment:
  ```txt
  python3 -m venv venv
  ```
- Activate the environment:

    === ":material-apple: macOS"
        ```txt
        . venv/bin/activate
        ```
    === ":material-linux: Linux"
        ```txt
        . venv/bin/activate
        ```
    === ":fontawesome-brands-windows: Windows"
        ```txt
        . venv/Scripts/activate      # Windows with Unix-like shells
        .\venv\Scripts\activate.bat  # Windows with CMD
        .\venv\Scripts\activate.ps1  # Windows with PowerShell
        ```

- The terminal should now show `(venv)`, indicating that the virtual environment
  is active.

<!-- prettier-ignore-end -->

#### Deactivating and Removing the Virtual Environment:

<!-- prettier-ignore-start -->

- Exit the environment:
  ```txt
  deactivate
  ```
- If this environment is no longer needed, delete it:

    === ":material-apple: macOS"
        ```txt
        rm -rf venv
        ```
    === ":material-linux: Linux"
        ```txt
        rm -rf venv
        ```
    === ":fontawesome-brands-windows: Windows"
        ```txt
        rmdir /s /q venv
        ```

<!-- prettier-ignore-end -->

### Installing from PyPI { #from-pypi }

EasyScience is available on **PyPI (Python Package Index)** and can be
installed using `pip`. To do so, use the following command:

```txt
pip install easyscience
```

To install a specific version of EasyScience, e.g., 1.0.3:

```txt
pip install 'easyscience==1.0.3'
```

To upgrade to the latest version:

```txt
pip install --upgrade easyscience
```

To upgrade to the latest version and force reinstallation of all
dependencies (useful if files are corrupted):

```txt
pip install --upgrade --force-reinstall easyscience
```

To check the installed version:

```txt
pip show easyscience
```

### Installing from GitHub <small>alternative</small> { #from-github data-toc-label="Installing from GitHub" }

Installing unreleased versions is generally not recommended but may be
useful for testing.

To install EasyScience from the `develop` branch of GitHub, for example:

```txt
pip install git+https://github.com/easyscience/core@develop
```

To include extra dependencies (e.g., dev):

```txt
pip install 'easyscience[dev] @ git+https://github.com/easyscience/core@develop'
```

## How to Run Tutorials

EasyScience includes a collection of **Jupyter Notebook examples** that
demonstrate key functionality. These tutorials serve as **step-by-step
guides** to help users understand the data analysis workflow. They are
available as **static HTML pages** in the
[:material-school: Tutorials](../tutorials/index.md) section.

In the next sections, we explain how to set up Jupyter and run the
tutorials interactively in two different ways: locally or online via
Google Colab.

If you decide to run the tutorials locally, you need to download them
first. This can be done individually via the :material-download:
**Download Notebook** button available on each tutorial page, or all at
once using the command line, as shown below.

### Run Tutorials Locally with Pixi <small>recommended</small> { #running-with-pixi data-toc-label="Run Tutorials Locally with Pixi" }

- Navigate to your existing Pixi project, created as described in the
  [Installing with Pixi](#installing-with-pixi) section.
- Add JupyterLab, Interactive Python shell and the Pixi kernel for
  Jupyter:
  ```txt
  pixi add --pypi jupyterlab ipython pixi-kernel
  ```
- Download all the EasyScience tutorials to the `tutorials/` directory.
- Start the JupyterLab server in the `tutorials/` directory to access
  the notebooks:
  ```txt
  pixi run jupyter lab tutorials/
  ```
- Your web browser should open automatically. Click on one of the
  `*.ipynb` files and select the `Python (Pixi)` kernel to get started.

### Classical Run Tutorials Locally

- Install Jupyter Notebook, Interactive Python shell and the IPython
  kernel:
  ```txt
  pip install notebook ipython ipykernel
  ```
- Add the virtual environment as a Jupyter kernel:
  ```txt
  python -m ipykernel install --user --name=venv --display-name "EasyScience Python kernel"
  ```
- Download all the EasyScience tutorials to the `tutorials/` directory.
- Start the Jupyter Notebook server in the `tutorials/` directory to
  access the notebooks:
  ```txt
  jupyter notebook tutorials/
  ```
- Your web browser should open automatically. Click on one of the
  `*.ipynb` files and select the `EasyScience Python kernel` to get
  started.

### Run Tutorials via Google Colab

**Google Colab** lets you run Jupyter Notebooks in the cloud without any
local installation. This is the fastest way to start experimenting with
EasyScience.

- Ensure you have a **Google account**.
- Go to the **[:material-school: Tutorials](../tutorials/index.md)**
  section.
- Click the :google-colab: **Open in Google Colab** button on any
  tutorial.
