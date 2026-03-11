# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:00:52 2026

@author: lfval
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUTS    = ROOT_DIR / "outputs"

for folder in [OUTPUTS]:
    folder.mkdir(parents=True, exist_ok=True)