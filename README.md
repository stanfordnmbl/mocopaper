Moco paper
==========

This repository holds the code and text for generating the manuscript on 
OpenSim Moco, a software toolkit for solving optimal control problems with
OpenSim musculoskeletal models.

The files in this repository are intended to be used through a Docker
container; see the Dockerfile. Using Docker ensures the results are 
reproducible. It is possible to run these files on Windows and Mac, but we
do not provide instructions.

The **code** folder contains Python scripts to generate results for the
manuscript.

The **paper** folder holds LaTeX source for generating a PDF of the manuscript.

The **resources** folder contains models and data used by the code.

The **results** and **figures** folders hold numerical results and figures
used in the paper. You must run the Docker container to obtain the results
and figures; we do not commit the results to the repository.
