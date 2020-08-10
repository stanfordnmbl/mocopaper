import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl
import os
import opensim as osim

import utilities
from squat_to_stand import SquatToStand
from tracking_walking import MotionTrackingWalking

def get_convergence_results(root_dir, result):
    metadata = result.convergence_metadata()
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
    return x, costs

def report_convergence(root_dir):

    fig = pl.figure(figsize=(5.2, 2.7))

    # tracking-walking
    # ----------------
    x, costs = get_convergence_results(root_dir, MotionTrackingWalking())
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('MocoTrack')
    ax.semilogx(x, costs, linestyle='', marker='o')
    ax.set_xlabel('number of mesh intervals')

    # squat-to-stand
    # --------------
    x, costs = get_convergence_results(root_dir, SquatToStand())
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('squat-to-stand')
    ax.semilogx(x, costs, linestyle='', marker='o')
    ax.set_xlabel('number of mesh intervals')
    ax.set_ylim(1.0, 1.7)


    fig.tight_layout()
    utilities.savefig(fig, os.path.join(root_dir, 'figures', 'S3_Fig'))
