import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl

import opensim as osim

from analytic import Analytic
from suspended_mass import SuspendedMass
from tracking_walking import MotionTrackingWalking
from squat_to_stand import SquatToStand

# TODO: create a docker container for these results and generating the preprint.
# TODO fix shift
# TODO: Add a periodicity cost to walking.
# TODO: Docker container gives very different result.
#       for suspended mass.


if __name__ == "__main__":
    import argparse

    results = {
        'analytic': Analytic(),
        'suspended-mass': SuspendedMass(),
        'tracking-walking': MotionTrackingWalking(),
        # 'predicting-walking': MotionPredictionAndAssistanceWalking(),
        'squat-to-stand': SquatToStand(),
   }

    parser = argparse.ArgumentParser(description="Generate results for the"
                                                 "OpenSim Moco publication.")
    parser.add_argument('--no-generate', dest='generate', action='store_false',
                        help='Skip generating the results; only report.')

    results_help = 'Names of results to generate or report ('
    for i, result_name in enumerate(results.keys()):
        results_help += result_name
        if i < len(results) - 1:
            results_help += ', '
        results_help += ').'

    parser.add_argument('--results', type=str, nargs='+', help=results_help)

    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="Passed to each result's generate_results() "
                             "and report_results() methods.")

    parser.set_defaults(generate=True)

    args = parser.parse_args()

    for requested in args.results:
        if not requested in results.keys():
            raise RuntimeError(f"Result {requested} not recognized.")
    for result_name, result_object in results.items():
        if args.results is None or result_name in args.results:
            if args.generate:
                print(f'Generating {result_name} results.')
                result_object.generate_results(args.args)
            print(f'Reporting {result_name} results.')
            result_object.report_results(args.args)

