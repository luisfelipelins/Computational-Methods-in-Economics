# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:00:00 2026

@author: lfval
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUTS    = ROOT_DIR / "outputs"

for folder in [OUTPUTS]:
    folder.mkdir(parents=True, exist_ok=True)
