## Setup
NOTE: All makefiles are designed to be run in an environment that supports bash.

# Install Dependencies
This package requires `make`, `python` (>=3.8), and (`conda`) to be installed. Make sure you have `conda` in your `PATH` (make sure you ran `conda init` according to `conda init -h`, e.g. `conda init zsh` or `conda init bash`). You'll need to run `make` from a `bash`-compatible shell (e.g. Terminal or Git Bash)

# Setup Development Environment:
Run:
`make init`
Will perform `make clean` first. If you want to only perform specific parts of the setup, look in the `init` routine of `makefile`.

If you want to make sure you're on the right virtual environment in your shell, run `make activate` and **then run the command it tells you to run**.

**Note:** You can run `make help` to see all available project commands.

## Develop

- If you want to install pip packages, add them to the pip dependencies in`pip_requirements.txt`, then run `make update`.
- If you want to install conda dependencies, add the install commands to `make update` in `makefile`, then run `make update`.

*Note*: As a dev, you'll likely want to `pip install pyls-mypy` so that your IDE 
can lint mypy (type annotations) via the Python Language Server.


### Known good development and build environment:

**OS:** MacOS

**Editor:** VSCode

**Linting:** `mypy` on VSCode using the [mypy](https://marketplace.visualstudio.com/items?itemName=matangover.mypy) extension installed. Be careful with the way the python language server is set up. Some linter setups won't find and lint imported python packages for typing (Spyder <=4.2.0 and vscode with pylance both had this problem). This setup does.

# Update Dependencies:
`make update`

# Run application:
`make run`

# Teardown and cleanup development environment (before reinitializing):
`make clean`

## Build
*TODO*