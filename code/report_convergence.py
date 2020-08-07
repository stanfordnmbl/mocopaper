import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl
import os
import opensim as osim

import utilities
from squat_to_stand import SquatToStand

def report_convergence(root_dir):

    # Squat-to-stand
    # --------------
    metadata = SquatToStand().convergence_metadata()

    x = list()
    costs = list()
    for md in metadata:
        solution_fpath = os.path.join(root_dir, 'results', 'convergence',
                                      md['solution_file'])
        table = osim.TimeSeriesTable(solution_fpath)
        num_mesh_intervals = md['num_mesh_intervals']
        # All problems use Hermite-Simpson transcription.
        if table.getNumRows() != 2 * num_mesh_intervals + 1:
            raise Exception("Inconsistent number of mesh intervals.")
        # TODO: num_mesh_intervals = (table.getNumRows() - 1) / 2
        x.append(num_mesh_intervals)
        costs.append(float(table.getTableMetaDataAsString('objective')))

    fig = pl.figure()
    ax = fig.add_subplot(1, 3, 3)
    ax.semilogx(x, costs, linestyle='', marker='o')
    ax.set_ylim(1.0, 1.7)
    utilities.savefig(fig, os.path.join(root_dir, 'figures', 'S3_Fig'))