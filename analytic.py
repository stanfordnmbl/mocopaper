import numpy as np

import opensim as osim

from moco_paper_result import MocoPaperResult

class Analytic(MocoPaperResult):
    def __init__(self):
        self.solution_file = 'results/analytic_solution.sto'
    def generate_results(self):
        model = osim.Model()
        body = osim.Body("b", 1, osim.Vec3(0), osim.Inertia(0))
        model.addBody(body)

        joint = osim.SliderJoint("j", model.getGround(), body)
        joint.updCoordinate().setName("coord")
        model.addJoint(joint)

        damper = osim.SpringGeneralizedForce("coord")
        damper.setViscosity(-1.0)
        model.addForce(damper)

        actu = osim.CoordinateActuator("coord")
        model.addForce(actu)
        model.finalizeConnections()

        moco = osim.MocoStudy()
        problem = moco.updProblem()

        problem.setModel(model)
        problem.setTimeBounds(0, 2)
        problem.setStateInfo("/jointset/j/coord/value", [-10, 10], 0, 5)
        problem.setStateInfo("/jointset/j/coord/speed", [-10, 10], 0, 2)
        problem.setControlInfo("/forceset/coordinateactuator", [-50, 50])

        problem.addGoal(osim.MocoControlGoal("effort", 0.5))

        solver = moco.initCasADiSolver()
        solver.set_num_mesh_intervals(50)
        solution = moco.solve()
        solution.write(self.solution_file)

    def report_results(self):
        solution = osim.MocoTrajectory(self.solution_file)
        time = solution.getTimeMat()

        exp = np.exp
        A = np.array([[-2 - 0.5 * exp(-2) + 0.5 * exp(2),
                       1 - 0.5 * exp(-2) - 0.5 * exp(2)],
                      [-1 + 0.5 * exp(-2) + 0.5 * exp(2),
                       0.5 * exp(-2) - 0.5 * exp(2)]])
        b = np.array([5, 2])
        c = np.linalg.solve(A, b)
        c2 = c[0]
        c3 = c[1]
        def x0_func(t):
            return (c2 * (-t - 0.5 * exp(-t) + 0.5 * exp(t)) +
                    c3 * (1 - 0.5 * exp(-t) - 0.5 * exp(t)))
        def x1_func(t):
            return (c2 * (-1 + 0.5 * exp(-t) + 0.5 * exp(t)) +
                    c3 * (0.5 * exp(-t) - 0.5 * exp(t)))
        expected_states = np.empty((len(time), 2))
        for itime in range(len(time)):
            expected_states[itime, 0] = x0_func(time[itime])
            expected_states[itime, 1] = x1_func(time[itime])

        states = solution.getStatesTrajectoryMat()
        square = np.sum((states - expected_states)**2, axis=1)
        mean = np.trapz(square, x=time) / (time[-1] - time[0])
        root = np.sqrt(mean)
        rms = root
        print(f'root-mean-square error in states: {rms}')
        with open('results/analytic_rms.txt', 'w') as f:
            f.write(f'{rms:.4f}')

