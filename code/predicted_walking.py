import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities

class MotionPredictedWalking(MocoPaperResult):
    def __init__(self):
        self.side = 'r'

    def configure_problem(self, problem, model):
        symmetry = osim.MocoPeriodicityGoal("symmetry")
        # Symmetric coordinate values (except for pelvis_tx) and speeds.
        for coord in model.getComponentsList():
            if not type(coord) is osim.Coordinate: continue
            if coord.getName().endswith("_r"):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0),
                    coord.getStateVariableNames().get(0).replace("_r", "_l")))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1),
                    coord.getStateVariableNames().get(1).replace("_r", "_l")))
            elif coord.getName().endswith("_l"):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0),
                    coord.getStateVariableNames().get(0).replace("_l", "_r")))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1),
                    coord.getStateVariableNames().get(1).replace("_l", "_r")))
            elif not coord.getName().endswith("_tx"):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0)))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1)))
        symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
            "/jointset/groundPelvis/pelvis_tx/speed"))
        # Symmetric coordinate actuator controls.
        symmetry.addControlPair(osim.MocoPeriodicityGoalPair("/lumbarAct"))
        # Symmetric muscle activations.
        for muscle in model.getComponentsList():
            if not muscle.getConcreteClassName().endswith('Muscle'): continue
            if muscle.getName().endswith("_r"):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    muscle.getStateVariableNames().get(0),
                    muscle.getStateVariableNames().get(0).replace("_r", "_l")))
            elif muscle.getName().endswith("_l"):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    muscle.getStateVariableNames().get(0),
                    muscle.getStateVariableNames().get(0).replace("_l", "_r")))
        problem.addGoal(symmetry)

        pi = np.pi
        problem.setStateInfo("/jointset/groundPelvis/pelvis_tilt/value",
                             [-20 * pi / 180, -10 * pi / 180])
        problem.setStateInfo("/jointset/groundPelvis/pelvis_tx/value",
                             [0, 1])
        problem.setStateInfo("/jointset/groundPelvis/pelvis_ty/value",
                             [0.75, 1.25])
        problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value",
                             [-10 * pi / 180, 60 * pi / 180])
        problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value",
                             [-10 * pi / 180, 60 * pi / 180])
        problem.setStateInfo("/jointset/knee_l/knee_angle_l/value",
                             [-50 * pi / 180, 0])
        problem.setStateInfo("/jointset/knee_r/knee_angle_r/value",
                             [-50 * pi / 180, 0])
        problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value",
                             [-15 * pi / 180, 25 * pi / 180])
        problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value",
                             [-15 * pi / 180, 25 * pi / 180])
        problem.setStateInfo("/jointset/lumbar/lumbar/value",
                             [0, 20 * pi / 180])

    def parse_args(self, args):
        self.skip_tracking = False
        self.skip_predicted = False
        self.visualize = False
        if len(args) == 0: return
        print('Received arguments {}'.format(args))
        if 'skip-tracking' in args:
            self.skip_tracking = True
        if 'skip-predicted' in args:
            self.skip_predicted = True
        if 'visualize' in args:
            self.visualize = True

    def generate_results(self, root_dir, args):
        self.parse_args(args)

        track = osim.MocoTrack()
        track.setName("motion_predicted_tracking")

        modelProcessor = osim.ModelProcessor(
            os.path.join(root_dir, "resources/Falisse2019/2D_gait.osim"))
        track.setModel(modelProcessor)
        coords_fpath = os.path.join(root_dir,
                                    "resources/Falisse2019/"
                                    "referenceCoordinates.sto")
        tableProcessor = osim.TableProcessor(coords_fpath)
        tableProcessor.append(osim.TabOpLowPassFilter(6))
        track.setStatesReference(tableProcessor)
        track.set_states_global_tracking_weight(10.0)
        track.set_allow_unused_references(True)
        track.set_track_reference_position_derivatives(True)
        track.set_apply_tracked_states_to_guess(True)
        track.set_initial_time(0.0)
        track.set_final_time(0.47008941)
        moco = track.initialize()
        moco.set_write_solution(os.path.join(root_dir, "results/"))
        problem = moco.updProblem()

        model = modelProcessor.process()
        model.initSystem()

        self.configure_problem(problem, model)

        # Effort. Get a reference to the MocoControlGoal that is added to every
        # MocoTrack problem by default.
        effort = problem.updGoal("control_effort")
        effort.setWeight(10)

        # Configure the solver.
        # =====================
        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.set_num_mesh_intervals(50)
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        solver.set_optim_max_iterations(1000)

        # Solve problem.
        # ==============
        moco.printToXML("motion_predicted_tracking.omoco")
        if not self.skip_tracking:
            trackingSolution = moco.solve()
            trackingSolution.write(os.path.join(
                root_dir,
                "results/motion_predicted_tracking_solution.sto"))
            trackingSolutionFull = osim.createPeriodicTrajectory(trackingSolution)
            trackingSolutionFull.write(os.path.join(
                root_dir,
                "results/motion_predicted_tracking_solution_fullcycle.sto"))


        # Prediction
        # ==========
        moco = osim.MocoStudy()
        moco.setName("motion_predicted_predicted")
        moco.set_write_solution(os.path.join(root_dir, "results/"))

        problem = moco.updProblem()
        modelProcessor = osim.ModelProcessor(os.path.join(
            root_dir, "resources/Falisse2019/2D_gait.osim"))
        problem.setModelProcessor(modelProcessor)

        model = modelProcessor.process()
        model.initSystem()

        self.configure_problem(problem, model)

        # Prescribed average gait speed.
        speedGoal = osim.MocoAverageSpeedGoal("speed")
        speedGoal.set_desired_average_speed(1.2)
        problem.addGoal(speedGoal)
        # Effort over distance.
        effortGoal = osim.MocoControlGoal("effort", 10)
        effortGoal.setExponent(3)
        effortGoal.setDivideByDisplacement(True)
        problem.addGoal(effortGoal)

        problem.setTimeBounds(0, [0.4, 0.6])

        solver = moco.initCasADiSolver()
        solver.set_num_mesh_intervals(50)
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        solver.set_optim_max_iterations(1000)
        # Use the solution from the tracking simulation as initial guess.
        # solver.setGuess(trackingSolution)
        solver.setGuessFile(os.path.join(
            root_dir,
            "results/motion_predicted_tracking_solution.sto"))

        moco.printToXML("motion_predicted_predicted.omoco")
        if not self.skip_predicted:
            predictedSolution = moco.solve()
            predictedSolution.write(
                os.path.join(root_dir,
                             "results",
                             "motion_predicted_predicted_solution.sto"))
            predictedSolutionFull = osim.createPeriodicTrajectory(
                predictedSolution)
            predictedSolutionFull.write(
                os.path.join(root_dir,
                             "results",
                             "motion_predicted_predicted_solution_fullcycle"
                             ".sto"))


        # Assisted
        # ========
        moco.setName("motion_predicted_assisted")
        device_r = osim.CoordinateActuator('ankle_angle_r')
        device_r.setName('device_r')
        device_r.setMinControl(-1)
        device_r.setMaxControl(0)
        device_r.set_optimal_force(1000)
        model.addComponent(device_r)

        device_l = osim.CoordinateActuator('ankle_angle_l')
        device_l.setName('device_l')
        device_l.setMinControl(-1)
        device_l.setMaxControl(0)
        device_l.set_optimal_force(1000)
        model.addComponent(device_l)

        problem.setModelProcessor(osim.ModelProcessor(model))

        symmetry = osim.MocoPeriodicityGoal.safeDownCast(
            problem.updGoal("symmetry"))
        symmetry.addControlPair(
            osim.MocoPeriodicityGoalPair("/device_l", "/device_r"))

        # TODO must set guess properly?
        # guess = solver.createGuess()
        # guess.insertStatesTrajectory(...)
        # guess.insertControlsTrajectory(...)
        solver.clearGuess()

        # moco.printToXML("motion_predicted_assisted.omoco")
        # assistedSolution = moco.solve()
        # assistedSolutionFull = osim.createPeriodicTrajectory(assistedSolution)
        # assistedSolutionFull.write("results/motion_predicted_assisted_solution_fullcycle.sto")
        # moco.visualize(full)


    def report_results(self, root_dir, args):
        self.parse_args(args)

        fig = plt.figure(figsize=(5.5, 5.5))
        gs = gridspec.GridSpec(9, 2)

        exp_half = osim.MocoTrajectory(
            os.path.join(root_dir, "resources", "Falisse2019",
                         "referenceCoordinates.sto"))
        exp = osim.createPeriodicTrajectory(exp_half)
        time_exp = exp.getTimeMat()
        pgc_exp = 100.0 * (time_exp - time_exp[0]) / (time_exp[-1] - time_exp[0])

        sol_track = osim.MocoTrajectory(
            os.path.join(root_dir,
                         "results",
                         "motion_predicted_tracking_solution_fullcycle.sto"))
        time_track = sol_track.getTimeMat()
        pgc_track = 100.0 * (time_track - time_track[0]) / (
                time_track[-1] - time_track[0])

        sol_predict = osim.MocoTrajectory(
            os.path.join(root_dir,
                         "results",
                         "motion_predicted_predicted_solution_fullcycle.sto"))
        time_predict = sol_predict.getTimeMat()
        pgc_predict = 100.0 * (time_predict - time_predict[0]) / (
                time_predict[-1] - time_predict[0])

        def toarray(simtk_vector):
            array = np.empty(simtk_vector.size())
            for i in range(simtk_vector.size()):
                array[i] = simtk_vector[i]
            return array

        coords = [
            (f'/jointset/hip_{self.side}/hip_flexion_{self.side}',
             'hip flexion', 1.0),
            (f'/jointset/knee_{self.side}/knee_angle_{self.side}',
             'knee flexion', -1.0),
            (f'/jointset/ankle_{self.side}/ankle_angle_{self.side}',
             'ankle plantarflexion', 1.0),
        ]
        for ic, coord in enumerate(coords):
            ax = plt.subplot(gs[(3 * ic):(3 * (ic + 1)), 0])

            y_exp = coord[2] * np.rad2deg(
                exp.getStateMat(f'{coord[0]}/value'))
            ax.plot(pgc_exp, y_exp, label='reference', color='gray')

            y_track = coord[2] * np.rad2deg(
                sol_track.getStateMat(f'{coord[0]}/value'))
            ax.plot(pgc_track, y_track, label='track', color='k',
                    linestyle='--')

            y_predict = coord[2] * np.rad2deg(
                sol_predict.getStateMat(f'{coord[0]}/value'))
            ax.plot(pgc_predict, y_predict, label='predict', color='blue',
                    linestyle='--')

            ax.set_xlim(0, 100)
            if ic == 1:
                ax.legend(frameon=False)
            if ic < len(coords) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            ax.set_ylabel(f'{coord[1]} (degrees)')

            ax.spines['bottom'].set_position('zero')
            utilities.publication_spines(ax)

        # TODO: Compare to EMG.
        muscles = [
            ('glut_max', 'glutes'),
            ('iliopsoas', 'iliopsoas'),
            ('hamstrings', 'hamstrings'),
            ('rect_fem', 'rectus femoris'),
            ('bifemsh', 'biceps femoris short head'),
            ('vasti', 'vasti'),
            ('gastroc', 'gastrocnemius'),
            ('soleus', 'soleus'),
            ('tib_ant', 'tibialis anterior'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[im, 1])
            activation_path = f'/{muscle[0]}_{self.side}/activation'
            ax.plot(pgc_track, sol_track.getStateMat(activation_path),
                    label='track', color='k', linestyle='--')
            ax.plot(pgc_predict, sol_predict.getStateMat(activation_path),
                    label='predict', color='blue', linestyle='--')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            if im < len(muscles) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            ax.set_title(muscle[1], fontsize=8)

            utilities.publication_spines(ax)

        fig.tight_layout(h_pad=0.05)

        fig.savefig(
            os.path.join(root_dir, 'figures', 'motion_predicted_walking.png'),
            dpi=600)

        # TODO: Plot model-generated GRFs.
