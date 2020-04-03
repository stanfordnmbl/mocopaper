import os
import numpy as np
from scipy.optimize import bisect

import opensim as osim

from moco_paper_result import MocoPaperResult

class LinearTangentSteering(MocoPaperResult):
    """In the "linear tangent steering" problem, we control the direction to
    apply
    a constant thrust to a point mass to move the mass a given vertical distance
    and maximize its final horizontal speed. This problem is described in
    Section 2.4 of Bryson and Ho [1].
    Bryson, A. E., Ho, Y.‐C., Applied Optimal Control, Optimization, Estimation,
    and Control. New York‐London‐Sydney‐Toronto. John Wiley & Sons. 1975.
    """

    def __init__(self):
        self.solution_file = '%s/results/linear_tangent_steering_solution.sto'
        self.a = 5.0
        self.final_time = 1.0
        self.final_height = 1.0
    def generate_results(self, root_dir, args):
        study = osim.MocoStudyFactory.createLinearTangentSteeringStudy(
            self.a, self.final_time, self.final_height)
        solution = study.solve()
        solution.write(self.solution_file % root_dir)

    def report_results(self, root_dir, args):
        solution = osim.MocoTrajectory(self.solution_file % root_dir)

        def residual(angle):
            secx = 1.0 / np.cos(angle)
            tanx = np.tan(angle)
            h = self.final_height
            T = self.final_time
            return (1.0 / np.sin(angle) -
                    np.log((secx + tanx) / (secx - tanx)) / (2.0 * tanx ** 2) -
                    4 * h / (self.a * T ** 2))
        initial_angle = bisect(residual, 0.01, 0.99 * 0.5 * np.pi)
        tan_initial_angle = np.tan(initial_angle)
        c = 2 * tan_initial_angle / self.final_time
        def txvalue(angle):
            seci = 1.0 / np.cos(initial_angle)
            tani = np.tan(initial_angle)
            secx = 1.0 / np.cos(angle)
            tanx = np.tan(angle)
            return self.a / c ** 2 * (seci - secx - tanx * np.log(
                (tani + seci) / (tanx + secx)))
        def tyvalue(angle):
            seci = 1.0 / np.cos(initial_angle)
            tani = np.tan(initial_angle)
            secx = 1.0 / np.cos(angle)
            tanx = np.tan(angle)
            return self.a / (2 * c ** 2) * (
                        (tani - tanx) * seci - (seci - secx) * tanx - np.log(
                    (tani + seci) / (tanx + secx)))
        def txspeed(angle):
            seci = 1.0 / np.cos(initial_angle)
            secx = 1.0 / np.cos(angle)
            tanx = np.tan(angle)
            return self.a / c * np.log(
                (tan_initial_angle + seci) / (tanx + secx))
        def tyspeed(angle):
            return self.a / c * (
                        1.0 / np.cos(initial_angle) - 1.0 / np.cos(angle))

        study = osim.MocoStudyFactory.createLinearTangentSteeringStudy(
            self.a, self.final_time, self.final_height)
        solver = study.initCasADiSolver()
        expected = solver.createGuess()
        txv = osim.Vector(expected.getNumTimes(), 0)
        tyv = osim.Vector(expected.getNumTimes(), 0)
        txs = osim.Vector(expected.getNumTimes(), 0)
        tys = osim.Vector(expected.getNumTimes(), 0)
        angleTraj = osim.Vector(expected.getNumTimes(), 0)
        for itime in range(expected.getNumTimes()):
            time = expected.getTime()[itime]
            angle = np.arctan(tan_initial_angle - c * time)
            angleTraj[itime] = angle
            txv[itime] = txvalue(angle)
            tyv[itime] = tyvalue(angle)
            txs[itime] = txspeed(angle)
            tys[itime] = tyspeed(angle)
        expected.setState('/jointset/tx/tx/value', txv)
        expected.setState('/jointset/ty/ty/value', tyv)
        expected.setState('/jointset/tx/tx/speed', txs)
        expected.setState('/jointset/ty/ty/speed', tys)
        expected.setControl('/forceset/actuator', angleTraj)

        rmse = expected.compareContinuousVariablesRMS(solution)

        print(f'root-mean-square error in solution: {rmse}')
        with open(os.path.join(root_dir,
                               'results/linear_tangent_steering_rms.txt'),
                  'w') as f:
            f.write(f'{rmse:.1e}')
