#!/bin/bash

python3 code/mocopaper.py

rubber --module=lualatex --pdf paper/MocoPaper.tex
