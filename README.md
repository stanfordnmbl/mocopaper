OpenSim Moco paper
==================

This repository holds the code and text for generating the manuscript on 
OpenSim Moco, a software toolkit for solving optimal control problems with
OpenSim musculoskeletal models.

View the Moco preprint at https://www.biorxiv.org/content/10.1101/839381v1.

To learn more about OpenSim Moco, visit https://simtk.org/projects/opensim-moco.

The **code** folder contains Python scripts to generate results for the
manuscript.

The **paper** folder holds LaTeX source for generating a PDF of the manuscript.

The **resources** folder contains models and data used by the code.

The **results** and **figures** folders hold numerical results and figures
used in the paper. You must run the Docker container to obtain the results
and figures; we do not commit the results to the repository.

You can use Docker to reproduce the results and paper. First, install Docker
on your computer. Then run the following command in your Terminal or 
Command Prompt:

    docker run --volume $(pwd):/output stanfordnmbl/mocopaper:preprint

The container takes about 3 hours to run.
The results, figures, and paper PDF will end up in your current directory.

If you want to run your own copy of the mocopaper repository instead of using
the copy of mocopaper stored inside the container, use the following command
instead:

    docker run --volume <local-mocopaper-repo>:/mocopaper stanfordnmbl/mocopaper:preprint

The results are saved to the results and figures folders of
`<local-mocopaper-repo>`, and the paper is saved to
`<local-mocopaper-repo>/paper/MocoPaper.pdf`.

You can view the Docker container for generating this paper at
https://hub.docker.com/repository/docker/stanfordnmbl/mocopaper.

It is possible to run these files on Windows and Mac, but we do not provide 
instructions.
