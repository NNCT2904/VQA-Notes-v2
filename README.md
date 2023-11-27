# QNN Barren Plateaus
This Jupyter Notebook project practices the experiment as given in the research design section in the project "Training Variational Quantum Models with Barren Plateaus Mitigation Strategies" ([see Jacob Cybulski's projects](http://jacobcybulski.com/) or [Thanh Nguyen's](https://nnct2904.github.io/projects)).

## Table of contents

- [QNN Barren Plateaus](#qnn-barren-plateaus)
  - [Table of contents](#table-of-contents)
- [Requirements](#requirements)
  - [For CONDA users](#for-conda-users)
- [Onboarding - Setup Project](#onboarding---setup-project)
  - [Clone this repository](#clone-this-repository)
  - [Install python dependencies](#install-python-dependencies)
- [Development - Launch the Notebook](#development---launch-the-notebook)
  - [Built in jupyter notebook and jupyter lab](#built-in-jupyter-notebook-and-jupyter-lab)
  - [Visual Studio Code](#visual-studio-code)

# Requirements
- [Python 3.10.4](https://www.python.org/downloads/release/python-3104/)
- [pip 22.0.4](https://pypi.org/project/pip/) for package management
- [pipenv](https://pipenv.pypa.io/en/latest/) for virtual environment and package management
- Optional: [Pyenv](https://github.com/pyenv/pyenv) for python version control

## For CONDA users
Install these packages:
- ipykernel, jupyter, jupyterlab, 
- matplotlib, numpy, pandas
- qiskit, qiskit-machine-learning, qiskit-ibm-runtime
- scikit-learn
- torch, torchvision, torchsummary
- python-dotenv

# Onboarding - Setup Project
We assume that you have the above requirements. Follow the instructions belows to install this python project.

## Clone this repository
Make sure that you have access to this repository, and your computer have the credentials associate with GitHub.
From your desired folder, for example, `/Documets/Code`, open a terminal and run:

```shell
git clone https://github.com/NNCT2904/VQA-Capacity-Notes.git
```

This command will download this repository to your computer as a folder named `VQA-Capacity-Notes`.

## Install python dependencies
You can install python 3.10.4 and set it as global python with `pyenv`:

```shell
# Install python 3.10.4
pyenv install 3.10.4

# Set 3.10.4 as global
pyenv global 3.10.4
```

After cloning the repository, we need to install nessessary packages to run this notebook. 
We are using `pipenv` as a virtual environment manager, and to install packages. 
Inside the project folder, open a terminal and run:

```shell
# This command will activate the virtual enviroment for this project.
pipenv shell

# This command install the packages listed in the Pipfile.
pipenv install
```

Yes, I am not using Conda, their way of environment management confuses me.

# Development - Launch the Notebook
Remember to use the correct python version and activate the environment first!

Run this command if you are not in the correct python version:
```shell
pyenv global 3.10.4
```

Run this command if you are not in the python environment:
```shell
# Inside the repository
pipenv shell
```

Create an `.env` file in the root folder to store IBMQ api key
```
TOKEN=paste your token here
HUB=your hub
GROUP=your group
PROJECT=your project
BACKEND=your backend
```

For example (not real token!): 
```
TOKEN=455cb6696
HUB=ibm-q-research
GROUP=deakin-uni-1
PROJECT=qnn-barren-plate
BACKEND=ibm_perth
```
## Built in jupyter notebook and jupyter lab
This repository also install the package `jupyter` and `jupyterlab`. The notebook should launch with this command:

```shell
jupyter notebook 
```

Or launch the Jupyter Lab with this command:
```shell
jupyter lab
```

## Visual Studio Code
You need to install the [vscode-jupyter](https://github.com/microsoft/vscode-jupyter) extension.
After opening the notebook with VSCode, you need to select the correct kernel. 
The name of the kernel shuld match with the environment name. 
To check the environment name, run the command:

```shell
pipenv --venv
```

<img src="./setupGuidelines/venv.png" width="100%">
<img src="./setupGuidelines/correctkernel.png" width="100%">
