# Computational Methods in Economics

This repository contains the codes used as solution to the problem sets of the graduate level subject of Computational Methods in Economics (CME) at FGV-EESP, 2026. These codes are authored by Luis Felipe Lins.

## Problem Set 1

In this problem set, our goal was to make estimates of the expected returns of Mega-Sena, the main lottery in Brazil. Detailed instructions can be found in the `PSet1 - Instructions.pdf` file, 
while the actual solution with the outputs and discussion can be found in the `PSet1 - CME - Luis Felipe Lins.pdf` file (review pending before PSet delivery).

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
