#!/bin/bash

python3 mocopaper.py

rubber --module=lualatex --pdf MocoPaper.tex
