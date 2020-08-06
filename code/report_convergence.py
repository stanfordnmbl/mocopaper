import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl
import os
import opensim as osim

import utilities
from squat_to_stand import SquatToStand

def report_convergence(root_dir):

    metadata = SquatToStand().convergence_metadata()

    num_mesh_intervals = list()
    costs = list()
    for md in metadata:
        num_mesh_intervals.append(md['num_mesh_intervals'])
        solution_fpath = os.path.join(root_dir, 'results', 'convergence',
                                      md['solution_file'])
        table = osim.TimeSeriesTable(solution_fpath)
        costs.append(float(table.getTableMetaDataAsString('objective')))

    fig = pl.figure()
    ax = fig.add_subplot(1, 3, 3)
    ax.semilogx(num_mesh_intervals, costs, linestyle='', marker='o')
    ax.set_ylim(1.0, 1.7)
    utilities.savefig(fig, os.path.join(root_dir, 'figures', 'S3_Fig'))