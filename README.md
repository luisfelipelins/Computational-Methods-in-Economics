# Computational Methods in Economics

This repository contains the codes used as solution to the problem sets of the graduate level subject of Computational Methods in Economics (CME) at FGV-EESP, 2026. These codes are authored by Luis Felipe Lins.

## Creating a venv

In each problem set, we'll have a separate `requirements.txt` file with the packages needed to run the codes and solve those problem sets. In this subsection, I describe in steps how to create a virtual environment, install the packages, and then run the codes inside the virtual environment on Windows (please check [this link](https://www.w3schools.com/python/python_virtualenv.asp) to see the analogous steps in other operating systems).

The first step after cloning the repository is to create the virtual environment. Open your command prompt (or Anaconda prompt, if you prefer), and navigate to the problem set's main folder (to navigate in cmd, check [https://www.geeksforgeeks.org/techtips/change-directories-in-command-prompt/](this) source). Once you're in this folder, run the following command to create the virtual environment:

`python -m venv venv`

Once the virtual environment is created, you need to navigate to the scripts folder inside it:

`cd venv/Script`

Inside the folder, run the command (within the command prompt) to activate the virtual environment:

(*) `activate`

Go back to the problem set's main folder. There you should find the `requirements.txt` file. It contains the packages and the versions of the packages used in the problem set's solution. To install it to the virtual environment, run the command:

`pip install -r requirements.txt`

After `pip` ends installing all the packages, the virtual environment should be ready to run the Python scripts. Go to any problem set's main folder, navigate to the code folder, and run the Python scripts (with the virtual environment active. If you already created the virtual environment, closed the command prompt, just go back to step (*) to activate it and proceed as follows). For example, to run the `config.py` file, just write in the command prompt:

`python config.py`

## Problem Set 1

In this problem set, our goal was to make estimates of the expected returns of Mega-Sena, the main lottery in Brazil. Detailed instructions can be found in the `PSet1 - Instructions.pdf` file, 
while the actual solution with the outputs and discussion can be found in the `PSet1 - CME - Luis Felipe Lins.pdf` file.

After creating the virtual environment and installing the required packages in the `requirements.txt` file, the codes in this sub-repository should be runned in the following order:

`config.py` $`\rightarrow`$ `create_dataset.py` $`\rightarrow`$ `analysis.py`

A brief description of each code's purpose below:

### `config.py`

Sets up the full working directory for this problem set.

### `create_dataset.py`

Creates the datasets used in this analysis (and also saves the two raw datasets). 

Mega-Sena results are directly downloaded from Loterias Caixa's website. 

Ticket price data comes from some news outlets reports on Mega-Sena ticket price readjustments:

- Poder360 (2014-2026): [https://web.archive.org/web/20250707005543/https://graficos.poder360.com.br/PEXGE/1/](https://web.archive.org/web/20250707005543/https://graficos.poder360.com.br/PEXGE/1/)
- Agora SP (2009-2014): [https://web.archive.org/web/20260302220800/https://agora.folha.uol.com.br/dicas/ult10107u620038.shtml](https://web.archive.org/web/20260302220800/https://agora.folha.uol.com.br/dicas/ult10107u620038.shtml)
- Extra Globo (check no adjustments were made between 2009 and 2014): [https://web.archive.org/web/20260302221013/https://extra.globo.com/economia-e-financas/jogos-da-mega-sena-quina-lotofacil-vao-ficar-ate-33-mais-caros-partir-de-maio-12234290.html](https://web.archive.org/web/20260302221013/https://extra.globo.com/economia-e-financas/jogos-da-mega-sena-quina-lotofacil-vao-ficar-ate-33-mais-caros-partir-de-maio-12234290.html)

### `analysis.py`

Performs the empirical analysis, as proposed in the description of this problem set.

## Problem Set 2

In this problem set, our goal is apply concepts of interpolation and root finding learned in topics 3 and 4 of the CME subject. Detailed instructions can be found in the `PSet2 - Instructions.pdf` file, 
while the actual solution with the outputs and discussion can be found in the `PSet2 - CME - Luis Felipe Lins.pdf` file.

After creating the virtual environment and installing the required packages in the `requirements.txt` file, the codes in this sub-repository should be runned in the following order:

`config.py` $`\rightarrow`$ `analysis.py`

A brief description of each code's purpose below:


### `config.py`

Sets up the full working directory for this problem set.

### `functions.py`

Defines the functions used in the `analysis.py`, such as bisection, secant, Newton-Raphson, etc.

### `analysis.py`

Performs the empirical analysis, as proposed in the description of this problem set.

## Problem Set 3

In this problem set, our goal is to explore core numerical methods in computational economics: optimization, numerical integration, and numerical differentiation. Detailed instructions can be found in the `PSet3 - Instructions.pdf` file, while the actual solution with the outputs and discussion can be found in the `PSet3 - CME - Luis Felipe Lins.pdf` file.

After creating the virtual environment and installing the required packages in the `requirements.txt` file, the codes in this sub-repository should be runned in the following order:

`config.py` $`\rightarrow`$ `analysis.py`

A brief description of each code's purpose below:


### `config.py`

Sets up the full working directory for this problem set.

### `functions.py`

Defines the functions used in the `analysis.py`, such as the objective functions, Gauss-Hermine, Monte Carlo, trapezoind method, etc.

### `analysis.py`

Performs the empirical analysis, as proposed in the description of this problem set.

## Final Work

In this final work, we estimate a general equilibrium quantitative model to study the redistibutive effects of increasing offshoring barriers (tariffs and institutional barriers). This final work is based on my thesis, whose main project can be found in the [following repository](https://github.com/luisfelipelins/The-Redistributive-Aspects-of-Trade-Barriers).

After creating the virtual environment and installing the required packages in the `requirements.txt` file, the codes in this sub-repository should be runned in the following order:

`config.py` $`\rightarrow`$ `estimation.py` $`\rightarrow`$ `analysis.py`

A brief description of each code's purpose below:


### `config.py`

Sets up the full working directory for this problem set.

### `functions.py`

Defines the functions used in the GeneralEquilibriumModel methods.

### `GeneralEquilibriumModel.py`

Defines GeneralEquilibriumModel class, used to estimate the models and create economy statistics.

### `estimation.py`

Solves the GeneralEquilibriumModel objects proposed in the final work, and saves them for further analysis. May take a few hours to run.

### `analysis.py`

Performs the analysis analysis, generating outputs.


