#!/bin/bash
# .ONESHELL:
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH:
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

PYTHON ?= python
PYTHONVERSION ?= 3.8.5
VENV_NAME ?= py_24789_machining_lstm

.PHONY = help init update clean activate run cloc

.DEFAULT_GOAL = help
all: help

help:
	@echo "---- 24789 ML for Machining Makefile ----"
	@echo "	make help: Information"
	@echo " "
	@echo "	make init: Clean and resetup the project on a *new* system."
	@echo "	make update: Install any new dependencies and update existing dependencies."
	@echo " "
	@echo "	make clean: Teardown and Cleanup the Development Environment."
	@echo " "
	@echo " make activate: Tells you what command to run to activate the virtual environment"
	@echo " "
	@echo "	make run: Activate the right environment and run the application (in development mode)."
	@echo " "
	@echo "	make cloc: Count Lines Of Code (and comments) by language, excluding builds and libraries."
	@echo "----------------------------------"


####
# Setup the Development Environment:
####
init:
	# Cleanup first:
	make clean
	
	# Load the correct version of python and setup the virtual environment:
	conda create --name ${VENV_NAME} python=${PYTHONVERSION} --yes
	
	# Install pip dependencies:
	make update
	
####
# Activate the project's virtual environment:
####
activate:
	echo "TO ACTIVATE, RUN: conda activate ${VENV_NAME}"

# Same but for makefile internal use:
_activate_internal:
	# Activate environment:
	$(CONDA_ACTIVATE) ${VENV_NAME};

####
# Install any new dependencies and update existing dependencies:
# Add any conda install requirements here, make sure to end the line correctly with semicolon slash
####
update:
	($(CONDA_ACTIVATE) ${VENV_NAME}; \
	${PYTHON} -m pip install -r pip_requirements.txt)
	
####
# Teardown and Cleanup the Development Environment:
####
clean:
	# Remove virtual environment:
	conda env remove --name ${VENV_NAME}

####
# Run the Development Application:
####
run:
	($(CONDA_ACTIVATE) ${VENV_NAME}; \
	${PYTHON} __main__.py)

####
# Count Lines of Code (for fun):
####
cloc:
	cloc --exclude-dir=__pycache__,node_modules,.idea,.mypy_cache,.pytest_cache .