import os
import opensim as osim

from analytic import Analytic
from linear_tangent_steering import LinearTangentSteering
from suspended_mass import SuspendedMass
from prescribed_walking import MotionPrescribedWalking
from tracking_walking import MotionTrackingWalking
from squat_to_stand import SquatToStand

from collections import OrderedDict

# TODO: create a docker container for these results and generating the preprint.
# TODO fix shift
# TODO: Add a periodicity cost to walking.
# TODO: Docker container gives very different result.
#       for suspended mass.


if __name__ == "__main__":
    import argparse

    results = OrderedDict()
    results['analytic'] = Analytic()
    results['linear-tangent-steering'] = LinearTangentSteering()
    results['suspended-mass'] = SuspendedMass()
    results['prescribed-walking'] = MotionPrescribedWalking()
    results['tracking-walking'] = MotionTrackingWalking()
    # 'predicting-walking': MotionPredictionAndAssistanceWalking(),
    results['squat-to-stand'] = SquatToStand()

    examples = """
    Examples
    --------
    
    Generate and report all results except convergence analysis:
      mocopaper.py
    
    Generate and report squat-to-stand results:
      mocopaper.py --results squat-to-stand
    
    Report squat-to-stand results without generating them:
      mocopaper.py --no-generate --results squat-to-stand
    
    Run and report convergence analysis on all results:
      mocopaper.py --convergence
    
    Run convergence analysis on squat-to-stand result:
      mocopaper.py --convergence --results squat-to-stand
    
    Report convergence analysis across results:
      mocopaper.py --no-generate --convergence
    """

    parser = argparse.ArgumentParser(description="Generate results for the"
                                                 "OpenSim Moco publication.",
                                     epilog=examples)
    parser.add_argument('--no-generate', dest='generate', action='store_false',
                        help='Skip generating the results; only report.')
    parser.add_argument('--convergence', dest='convergence',
                        action='store_true',
                        help='Run and plot convergence analysis instead of '
                             'regular results.')

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

    print(f'OpenSim Moco {osim.GetMocoVersionAndDate()}')

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    if args.results:
        for requested in args.results:
            if not requested in results.keys():
                raise RuntimeError(f"Result {requested} not recognized.")
    for result_name, result_object in results.items():
        if args.results is None or result_name in args.results:
            if args.convergence:
                if args.generate:
                    print(f'Generating {result_name} convergence results.')
                    result_object.generate_convergence_results(root_dir,
                                                               args.args)
            else:
                if args.generate:
                    print(f'Generating {result_name} results.')
                    result_object.generate_results(root_dir, args.args)
                print(f'Reporting {result_name} results.')
                result_object.report_results(root_dir, args.args)
    if args.convergence:
        if not args.generate and not (args.results is None):
            raise Exception("If passing --convergence, cannot pass both "
                            "--no-generate and --results")
        # Only report convergence if `--results` was not passed.
        if args.results is None:
            import report_convergence
            report_convergence.report_convergence(root_dir)



