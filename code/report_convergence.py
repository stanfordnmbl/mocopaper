import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pylab as pl
import os
import opensim as osim

import utilities
from prescribed_walking import MotionPrescribedWalking
from tracking_walking import MotionTrackingWalking
from squat_to_stand import SquatToStand

def get_convergence_results(root_dir, result):
    metadata = result.convergence_metadata()
    x = list()
    costs = list()
    for md in metadata:
        solution_fpath = os.path.join(root_dir, 'results', 'convergence',
                                      md['solution_file'])
        if not os.path.exists(solution_fpath):
            print(f"Warning: solution {solution_fpath} does not exist.")
            continue
        table = osim.TimeSeriesTable(solution_fpath)
        num_mesh_intervals = md['num_mesh_intervals']
        # All problems use Hermite-Simpson transcription.
        if table.getNumRows() != 2 * num_mesh_intervals + 1:
            print("Warning: inconsistent number of mesh intervals "
                  f"({(table.getNumRows() - 1) / 2} vs {num_mesh_intervals}).")
        num_mesh_intervals = (table.getNumRows() - 1) / 2
        x.append(num_mesh_intervals)
        costs.append(float(table.getTableMetaDataAsString('objective')))
    # Normalize costs.
    if costs:
        costs = np.array(costs) / costs[-1]
    return x, costs

def report_convergence(root_dir):

    fig = pl.figure(figsize=(5.2, 2.5))
    ymax = 1.25
    yticks = [0, 1.0]

    # prescribed-walking
    # ------------------
    x, costs = get_convergence_results(root_dir, MotionPrescribedWalking())
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('MocoInverse\n(Fig 7, black)')
    ax.semilogx(x, costs, linestyle='', marker='.', color='black')
    ax.set_xlabel('number of mesh intervals')
    ax.set_yticks(yticks)
    ax.set_ylim(0, ymax)
    ax.set_ylabel('objective (normalized)')
    utilities.publication_spines(ax)

    # tracking-walking
    # ----------------
    x, costs = get_convergence_results(root_dir, MotionTrackingWalking())
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('MocoTrack\n(Fig 8)')
    ax.semilogx(x, costs, linestyle='', marker='.', color='black')
    ax.set_xlabel('number of mesh intervals')
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    utilities.publication_spines(ax)

    # squat-to-stand
    # --------------
    x, costs = get_convergence_results(root_dir, SquatToStand())
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Predictive MocoStudy\n(Fig 10, black)')
    ax.semilogx(x, costs, linestyle='', marker='.', color='black')
    ax.set_xlabel('number of mesh intervals')
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    utilities.publication_spines(ax)

    fig.tight_layout()
    utilities.savefig(fig, os.path.join(root_dir, 'figures', 'S3_Fig'))
