from abc import ABC, abstractmethod
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.lines import Line2D
import pylab as pl

import opensim as osim

# TODO: create a docker container for these results and generating the preprint.
# TODO fix shift
# TODO: Add a periodicity cost to walking.
# TODO: Docker container gives very different result.
#       for suspended mass.

import utilities

class MocoPaperResult(ABC):
    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass

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


class SuspendedMass(MocoPaperResult):
    width = 0.2
    xinit = -0.7 * width
    xfinal = 0.7 * width
    yinit = -width
    yfinal = -width + 0.05

    def __init__(self):
        pass
    def build_model(self):
        model = osim.ModelFactory.createPlanarPointMass()
        body = model.updBodySet().get("body")
        model.updForceSet().clearAndDestroy()
        model.finalizeFromProperties()

        model.updCoordinateSet().get('tx').setRangeMin(-self.width)
        model.updCoordinateSet().get('tx').setRangeMax(self.width)
        model.updCoordinateSet().get('ty').setRangeMin(-2 * self.width)
        model.updCoordinateSet().get('ty').setRangeMax(0)

        actuL = osim.DeGrooteFregly2016Muscle()
        actuL.setName("left")
        actuL.set_max_isometric_force(10)
        actuL.set_optimal_fiber_length(.20)
        actuL.set_tendon_slack_length(0.10)
        actuL.set_pennation_angle_at_optimal(0.0)
        actuL.set_ignore_tendon_compliance(True)
        actuL.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(-self.width, 0, 0))
        actuL.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuL)

        actuM = osim.DeGrooteFregly2016Muscle()
        actuM.setName("middle")
        actuM.set_max_isometric_force(10)
        actuM.set_optimal_fiber_length(0.09)
        actuM.set_tendon_slack_length(0.1)
        actuM.set_pennation_angle_at_optimal(0.0)
        actuM.set_ignore_tendon_compliance(True)
        actuM.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(0, 0, 0))
        actuM.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuM)

        actuR = osim.DeGrooteFregly2016Muscle()
        actuR.setName("right")
        actuR.set_max_isometric_force(20)
        actuR.set_optimal_fiber_length(.21)
        actuR.set_tendon_slack_length(0.09)
        actuR.set_pennation_angle_at_optimal(0.0)
        actuR.set_ignore_tendon_compliance(True)
        actuR.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(+self.width, 0, 0))
        actuR.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuR)

        model.finalizeConnections()

        return model

    def create_study(self):
        study = osim.MocoStudy()
        problem = study.updProblem()
        problem.setModel(self.build_model())
        problem.setStateInfo("/jointset/tx/tx/value", [-self.width, self.width],
                             self.xinit, self.xfinal)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * self.width, 0],
                             self.yinit, self.yfinal)
        problem.setStateInfo("/jointset/tx/tx/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/jointset/ty/ty/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.setControlInfo("/forceset/left", [0, 1])
        problem.setControlInfo("/forceset/middle", [0, 1])
        problem.setControlInfo("/forceset/right", [0, 1])

        solver = study.initCasADiSolver()
        solver.set_num_mesh_intervals(100)

        return study

    def predict(self, run_time_stepping=False, exponent=2):

        study = self.create_study()
        problem = study.updProblem()
        problem.setTimeBounds(0, [0.4, 0.8])

        effort = osim.MocoControlGoal("effort")
        effort.setExponent(exponent)
        problem.addGoal(effort)

        problem.addGoal(osim.MocoFinalTimeGoal("time", 0.01))

        solution = study.solve()

        # study.visualize(solution)

        time_stepping = None
        if run_time_stepping:
            time_stepping = osim.simulateIterateWithTimeStepping(solution,
                                                            self.build_model())

        return solution, time_stepping

    def track(self, prediction_solution, exponent=2):

        track = osim.MocoTrack()
        track.setName('suspended_mass_tracking')
        track.setModel(osim.ModelProcessor(self.build_model()))
        track.setStatesReference(
            osim.TableProcessor(prediction_solution.exportToStatesTable()))
        track.set_states_global_tracking_weight(10.0)
        track.set_allow_unused_references(True)
        # track.set_control_effort_weight(1.0)
        track.set_mesh_interval(0.003)

        study = track.initialize()
        problem = study.updProblem()
        problem.setStateInfo("/jointset/tx/tx/value", [-self.width, self.width],
                             self.xinit)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * self.width, 0],
                             self.yinit)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.updPhase().setDefaultSpeedBounds(osim.MocoBounds(-15, 15))
        tracking = osim.MocoStateTrackingGoal.safeDownCast(
            problem.updGoal('state_tracking'))
        tracking.setPattern('.*(value|speed)$')

        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal('control_effort'))
        effort.setExponent(exponent)

        solver = osim.MocoCasADiSolver.safeDownCast(
            study.updSolver())
        solver.set_optim_convergence_tolerance(-1)
        solver.set_optim_constraint_tolerance(-1)

        solution = study.solve()

        return solution

    def generate_results(self):
        predict_solution, time_stepping = self.predict(True)
        predict_solution.write('results/suspended_mass_prediction_solution.sto')
        time_stepping.write('results/suspended_mass_time_stepping.sto')
        track_solution = self.track(predict_solution)
        track_solution.write('results/suspended_mass_track_solution.sto')
        track_solution_p = self.track(predict_solution, 4)
        track_solution_p.write('results/suspended_mass_track_p_solution.sto')

    def report_results(self):
        pl.figure()
        fig = plt.figure(figsize=(5.2, 2.7))
        grid = gridspec.GridSpec(3, 4)
        predict_solution = osim.MocoTrajectory(
            'results/suspended_mass_prediction_solution.sto')
        time_stepping = osim.MocoTrajectory(
            'results/suspended_mass_time_stepping.sto')
        track_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_solution.sto')
        track_p_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_p_solution.sto')

        time_stepping_rms = time_stepping.compareContinuousVariablesRMSPattern(
            predict_solution,
            'states', '/jointset.*value$')
        print(f'time-stepping rms: {time_stepping_rms}')
        with open('results/suspended_mass_'
                  'time_stepping_coord_rms.txt', 'w') as f:
            f.write(f'{time_stepping_rms:.4f}')

        track_rms = track_solution.compareContinuousVariablesRMSPattern(
            predict_solution, 'states', '/forceset.*activation$')
        print(f'track rms: {track_rms}')
        with open('results/suspended_mass_'
                  'track_activation_rms.txt', 'w') as f:
            f.write(f'{track_rms:.4f}')

        track_p_rms = track_p_solution.compareContinuousVariablesRMSPattern(
            predict_solution, 'states', '/forceset.*activation$')
        print(f'track p=4 rms: {track_p_rms}')
        with open('results/suspended_mass_'
                  'track_p_activation_rms.txt', 'w') as f:
            f.write(f'{track_p_rms:.3f}')

        ax = fig.add_subplot(grid[:, 0:2])
        ax.plot([-self.width, self.xinit],
                [0, self.yinit], color='tab:red', alpha=0.3,
                linewidth=3)
        ax.plot([0, self.xinit],
                [0, self.yinit], color='tab:red', alpha=0.3,
                linewidth=3)
        ax.plot([self.width, self.xinit],
                [0, self.yinit], color='tab:red', alpha=0.3,
                linewidth=3)

        ax.plot([-self.width, self.xfinal],
                [0, self.yfinal], color='tab:red',
                linewidth=3)
        ax.plot([0, self.xfinal],
                [0, self.yfinal], color='tab:red',
                linewidth=3)
        ax.plot([self.width, self.xfinal],
                [0, self.yfinal], color='tab:red',
                linewidth=3)
        ax.plot([-1.1 * self.width, 1.1 * self.width], [0.005, 0.005], color='k',
                linewidth=3)

        ax.annotate('', xy=(0, -0.85 * self.width), xycoords='data',
                    xytext=(0, -0.7 * self.width),
                    arrowprops=dict(width=0.2, headwidth=2, headlength=1.5,
                                    facecolor='black'),
                    )
        ax.text(0, -0.78 * self.width, ' g')

        a = ax.plot(predict_solution.getStateMat('/jointset/tx/tx/value'),
                predict_solution.getStateMat('/jointset/ty/ty/value'),
                color='lightgray', linewidth=6,
                    label='prediction, $x^2$')
        b = ax.plot(time_stepping.getStateMat('/jointset/tx/tx/value'),
                time_stepping.getStateMat('/jointset/ty/ty/value'),
                linewidth=3.5,
                    label='time-stepping',
                    )
        c = ax.plot(track_solution.getStateMat('/jointset/tx/tx/value'),
                track_solution.getStateMat('/jointset/ty/ty/value'),
                linewidth=1.5,
                   label='MocoTrack, $x^2$')
        d = ax.plot(track_p_solution.getStateMat('/jointset/tx/tx/value'),
                track_p_solution.getStateMat('/jointset/ty/ty/value'),
                linestyle='-', linewidth=1,
                    label='MocoTrack, $x^4$')
        plt.annotate('initial', (self.xinit, self.yinit),
                     xytext=(self.xinit - 0.02, self.yinit - 0.03))
        plt.annotate('final', (self.xfinal, self.yfinal),
                     xytext=(self.xfinal + 0.008, self.yfinal - 0.03))
        ax.plot([self.xinit, self.xfinal],
                [self.yinit, self.yfinal],
                color='k', marker='o',
                linestyle='')
        ax.plot([0], [0.8 * self.width])
        scalebar = AnchoredSizeBar(ax.transData, 0.05, label='5 cm',
                                   loc='lower center',
                                   pad=-2.0,
                                   frameon=False)
        ax.add_artist(scalebar)
        ax.set_title('trajectory of point mass')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'datalim')
        ax.legend(frameon=False,
                  handlelength=1.9,
                  loc='upper center')
        utilities.publication_spines(ax)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


        ax = fig.add_subplot(grid[0, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/left/activation'),
                color='lightgray', linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/left/activation'),
                linewidth=3.5, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/left/activation'),
                linewidth=1.5, clip_on=False)
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/left/activation'),
                linestyle='-', linewidth=1, clip_on=False)
        plt.text(0.5, 1.08, 'left activation',
                 horizontalalignment='center',
                 transform=ax.transAxes)
        plt.grid(which='both', axis='y', clip_on=False)
        ax.set_yticks([0.5], minor=True)
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.autoscale(enable=True, axis='x', tight=True)
        utilities.publication_spines(ax)

        ax = fig.add_subplot(grid[1, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/middle/activation'),
                color='lightgray', linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/middle/activation'),
                linewidth=3.5, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/middle/activation'),
                linewidth=1.5, clip_on=False)
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/middle/activation'),
                linewidth=1, clip_on=False)
        plt.text(0.5, 1.08, 'middle activation',
                 horizontalalignment='center',
                 transform=ax.transAxes)
        plt.grid(which='both', axis='y', clip_on=False)
        ax.set_yticks([0.5], minor=True)
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.autoscale(enable=True, axis='x', tight=True)
        utilities.publication_spines(ax)

        ax = fig.add_subplot(grid[2, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/right/activation'),
                color='lightgray', linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/right/activation'),
                linewidth=3.5, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/right/activation'),
                linewidth=1.5, clip_on=False)
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/right/activation'),
                linewidth=1, clip_on=False)
        plt.text(0.5, 1.08, 'right activation',
                 horizontalalignment='center',
                 transform=ax.transAxes)
        plt.grid(which='both', axis='y', clip_on=False)
        ax.set_yticks([0.5], minor=True)
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('time (s)')
        utilities.publication_spines(ax)

        fig.tight_layout()

        pl.savefig('figures/suspended_mass.png', dpi=600)
        pl.savefig('figures/suspended_mass.pdf')

class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        self.initial_time = 0.450
        self.final_time = 1.565
        self.footstrike = 0.836 # 1.424
        self.mocotrack_solution_file = \
            'results/motion_tracking_walking_track_solution.sto'
        self.mocoinverse_solution_file = \
            'results/motion_tracking_walking_inverse_solution.sto'
        self.mocoinverse_jointreaction_solution_file = \
            'results/motion_tracking_walking_inverse_jointreaction_solution.sto'
        self.side = 'l'
        self.emg_sensor_names = [
            'SOL', 'GAS', 'TA', 'MH', 'BF', 'VL', 'VM', 'RF', 'GMAX', 'GMED'
        ]

    def shift(self, time, y, initial_time=None, final_time=None, starting_time=None):
        if not initial_time:
            initial_time = self.initial_time
        if not final_time:
            final_time = self.final_time
        if not starting_time:
            starting_time = self.footstrike
        return utilities.shift_data_to_cycle(initial_time, final_time,
                                   starting_time, time, y, cut_off=True)

    def create_model_processor(self):
        modelProcessor = osim.ModelProcessor(
            # "resources/ArnoldSubject02Walk3/subject02_armless.osim")
            "resources/Rajagopal2016/subject_walk_armless.osim")
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        # modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
        modelProcessor.append(osim.ModOpAddReserves(1, 20))
        modelProcessor.process().printToXML("subject_armless_for_cmc.osim")
        ext_loads_xml = "resources/Rajagopal2016/grf_walk.xml"
        modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))
        return modelProcessor

    def generate_results(self):
        # Create and name an instance of the MocoTrack tool.
        track = osim.MocoTrack()
        track.setName("motion_tracking_walking")

        modelProcessor = self.create_model_processor()
        track.setModel(modelProcessor)

        # TODO:
        #  - avoid removing muscle passive forces

        coordinates = osim.TableProcessor(
            "resources/Rajagopal2016/coordinates.sto")
        coordinates.append(osim.TabOpLowPassFilter(6))
        track.setStatesReference(coordinates)
        track.set_states_global_tracking_weight(0.05)

        # This setting allows extra data columns contained in the states
        # reference that don't correspond to model coordinates.
        track.set_allow_unused_references(True)

        track.set_track_reference_position_derivatives(True)

        # Initial time, final time, and mesh interval.
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(0.01)

        moco = track.initialize()
        moco.set_write_solution("results/")

        problem = moco.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal("control_effort"))

        model = modelProcessor.process()
        model.initSystem()
        forceSet = model.getForceSet()
        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            if 'pelvis' in str(forcePath):
                effort.setWeightForControl(forcePath, 10)

        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        # solver.set_optim_convergence_tolerance(1e-4)

        # Solve and visualize.
        moco.printToXML('motion_tracking_walking.omoco')
        # 45 minutes
        solution = moco.solve()
        solution.write(self.mocotrack_solution_file)
        # moco.visualize(solution)

        # tasks = osim.CMC_TaskSet()
        # for coord in model.getCoordinateSet():
        #     task = osim.CMC_Joint()
        #     task.setName(coord.getName())
        #     task.setCoordinateName(coord.getName())
        #     task.setKP(100, 1, 1)
        #     task.setKV(20, 1, 1)
        #     task.setActive(True, False, False)
        #     tasks.cloneAndAppend(task)
        # tasks.printToXML('motion_tracking_walking_cmc_tasks.xml')
        # TODO plotting should happen separately from generating the results.
        # cmc = osim.CMCTool()
        # cmc.setName('motion_tracking_walking_cmc')
        # cmc.setExternalLoadsFileName('grf_walk.xml')
        # # TODO filter:
        # cmc.setDesiredKinematicsFileName('coordinates.sto')
        # # cmc.setLowpassCutoffFrequency(6)
        # cmc.printToXML('motion_tracking_walking_cmc_setup.xml')
        cmc = osim.CMCTool('motion_tracking_walking_cmc_setup.xml')
        # 1 minute
        cmc.run()


        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.02)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_tolerance(1e-3)
        # 8 minutes
        solution = inverse.solve()
        solution.getMocoSolution().write(self.mocoinverse_solution_file)

        study = inverse.initialize()
        reaction_r = osim.MocoJointReactionGoal('reaction_r', 0.1)
        reaction_r.setJointPath('/jointset/walker_knee_r')
        reaction_r.setReactionMeasures(['force-x', 'force-y'])
        reaction_l = osim.MocoJointReactionGoal('reaction_l', 0.1)
        reaction_l.setJointPath('/jointset/walker_knee_l')
        reaction_l.setReactionMeasures(['force-x', 'force-y'])
        problem = study.updProblem()
        problem.addGoal(reaction_r)
        problem.addGoal(reaction_l)
        # The knee force is being generated by the reserve actuator.
        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal('excitation_effort'))
        effort.setWeightForControl(
            '/forceset/reserve_jointset_ankle_r_ankle_angle_r',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_ankle_l_ankle_angle_l',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_walker_knee_r_knee_angle_r',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_walker_knee_l_knee_angle_l',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_r_hip_flexion_r',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_l_hip_flexion_l',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_r_hip_adduction_r',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_l_hip_adduction_l',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_r_hip_rotation_r',
            50.0)
        effort.setWeightForControl(
            '/forceset/reserve_jointset_hip_l_hip_rotation_l',
            50.0)


        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        # TODO: try 1e-2 for MocoInverse without JR minimization.
        solver.set_optim_convergence_tolerance(1e-2)

        # 50 minutes.
        solution_reaction = study.solve()
        solution_reaction.write(self.mocoinverse_jointreaction_solution_file)

        # TODO: Minimize joint reaction load!
    def plot(self, ax, time, y, shift=True, fill=False, *args, **kwargs):
        if shift:
            shifted_time, shifted_y = self.shift(time, y)
        else:
            duration = self.final_time - self.initial_time
            shifted_time, shifted_y = self.shift(time, y,
                                                 starting_time=self.footstrike + 0.5 * duration)

        # TODO is this correct?
        duration = self.final_time - self.initial_time
        if fill:
            plt.fill_between(
                100.0 * shifted_time / duration,
                shifted_y,
                np.zeros_like(shifted_y),
                *args,
                clip_on=False, **kwargs)
        else:
            return ax.plot(100.0 * shifted_time / duration, shifted_y, *args,
                           clip_on=False, **kwargs)

    def load_electromyography(self):
        anc = utilities.ANCFile('resources/Rajagopal2016/emg_walk_raw.anc')
        raw = anc.data
        fields_to_remove = []
        for name in anc.names:
            if name != 'time' and name not in self.emg_sensor_names:
                fields_to_remove.append(name)
        del name

        # We don't actually use the data that is initially in this object. We
        # will overwrite all the data with the filtered data.
        filtered_emg = utilities.remove_fields_from_structured_ndarray(raw,
                fields_to_remove).copy()

        # Debugging.
        emg_fields = list(filtered_emg.dtype.names)
        emg_fields.remove('time')
        for expected_field in self.emg_sensor_names:
            if expected_field not in emg_fields:
                raise Exception("EMG field {} not found.".format(
                    expected_field))

        # Filter all columns.
        for name in filtered_emg.dtype.names:
            if name != 'time':
                scaled_raw = anc.ranges[name] * 2 / 65536.0 * 0.001 * anc[name]
                filtered_emg[name] = utilities.filter_emg(
                    scaled_raw.copy(), anc.rates[name])
                filtered_emg[name] /= np.max(filtered_emg[name])
        return filtered_emg

    def calc_reserves(self, solution):
        modelProcessor = self.create_model_processor()
        model = modelProcessor.process()
        output = osim.analyze(model, solution, ['.*reserve.*actuation'])
        return output

    def calc_max_knee_reaction_force(self, solution):
        modelProcessor = self.create_model_processor()
        model = modelProcessor.process()
        jr = osim.analyzeSpatialVec(model, solution,
                                    ['.*walker_knee.*reaction_on_parent.*'])
        jr = jr.flatten(['_mx', '_my', '_mz', '_fx', '_fy', '_fz'])
        max = -np.inf
        # traj = np.empty(jr.getNumRows())
        for itime in range(jr.getNumRows()):
            for irxn in range(int(jr.getNumColumns() / 6)):
                fx = jr.getDependentColumnAtIndex(6 * irxn + 3)[itime]
                fy = jr.getDependentColumnAtIndex(6 * irxn + 4)[itime]
                fz = jr.getDependentColumnAtIndex(6 * irxn + 5)[itime]
                norm = np.sqrt(fx**2 + fy**2 + fz**2)
                # traj[itime] = norm
                max = np.max([norm, max])
        g = np.abs(model.get_gravity()[1])
        state = model.initSystem()
        mass = model.getTotalMass(state)
        weight = mass * g
        return max / weight

    def report_results(self):

        sol_track_table = osim.TimeSeriesTable(self.mocotrack_solution_file)
        track_duration = sol_track_table.getTableMetaDataString('solver_duration')
        track_duration = float(track_duration) / 60.0 / 60.0
        print('track duration ', track_duration)
        with open('results/'
                  'motion_tracking_walking_track_duration.txt', 'w') as f:
            f.write(f'{track_duration:.1f}')

        sol_inverse_table = osim.TimeSeriesTable(self.mocoinverse_solution_file)
        inverse_duration = sol_inverse_table.getTableMetaDataString('solver_duration')
        inverse_duration = float(inverse_duration) / 60.0
        print('inverse duration ', inverse_duration)
        with open('results/'
                  'motion_tracking_walking_inverse_duration.txt', 'w') as f:
            f.write(f'{inverse_duration:.1f}')

        sol_inverse_jr_table = osim.TimeSeriesTable(self.mocoinverse_jointreaction_solution_file)
        inverse_jr_duration = sol_inverse_jr_table.getTableMetaDataString('solver_duration')
        inverse_jr_duration = float(inverse_jr_duration) / 60.0 / 60.0
        print('inverse joint reaction duration ', inverse_jr_duration)
        with open('results/'
                  'motion_tracking_walking_inverse_jr_duration.txt', 'w') as f:
            f.write(f'{inverse_jr_duration:.1f}')

        emg = self.load_electromyography()


        sol_track = osim.MocoTrajectory(self.mocotrack_solution_file)
        time_track = sol_track.getTimeMat()

        sol_inverse = osim.MocoTrajectory(self.mocoinverse_solution_file)
        time_inverse = sol_inverse.getTimeMat()

        sol_inverse_jointreaction = \
            osim.MocoTrajectory(self.mocoinverse_jointreaction_solution_file)
        sol_inverse_jointreaction.insertStatesTrajectory(
            sol_inverse.exportToStatesTable(), False)
        mocoinverse_jr_solution_file = \
            self.mocoinverse_jointreaction_solution_file.replace('.sto',
                                                                 '_with_q_u.sto')

        sol_inverse_jointreaction.write(mocoinverse_jr_solution_file)
        time_inverse_jointreaction = sol_inverse_jointreaction.getTimeMat()

        modelProcessor = self.create_model_processor()
        model = modelProcessor.process()
        model.initSystem()
        print(f'Degrees of freedom: {model.getCoordinateSet().getSize()}')


        # report = osim.report.Report(model, self.mocotrack_solution_file)
        # report.generate()
        #
        # report = osim.report.Report(model, self.mocoinverse_solution_file)
        # report.generate()

        # TODO: slight shift in CMC solution might be due to how we treat
        # percent gait cycle and the fact that CMC is missing 0.02 seconds.
        sol_cmc = osim.TimeSeriesTable('results/motion_tracking_walking_cmc_results/'
                                       'motion_tracking_walking_cmc_states.sto')
        # TODO uh oh overwriting the MocoInverse solution.
        # sol_inverse.setStatesTrajectory(sol_cmc)
        time_cmc = np.array(sol_cmc.getIndependentColumn())

        mocosol_cmc = sol_inverse.clone()
        mocosol_cmc.insertStatesTrajectory(sol_cmc, True)
        inv_sol_rms = \
            sol_inverse.compareContinuousVariablesRMSPattern(mocosol_cmc,
                                                             'states',
                                                             '.*activation')

        print('CMC MocoInverse activation RMS: ', inv_sol_rms)
        with open('results/motion_tracking_walking_'
                  'inverse_cmc_rms.txt', 'w') as f:
            f.write(f'{inv_sol_rms:.3f}')



        plot_breakdown = False

        if plot_breakdown:
            fig = utilities.plot_joint_moment_breakdown(model, sol_inverse,
                                        ['/jointset/hip_l/hip_flexion_l',
                                         '/jointset/hip_l/hip_adduction_l',
                                         '/jointset/hip_l/hip_rotation_l',
                                         '/jointset/walker_knee_l/knee_angle_l',
                                         '/jointset/ankle_l/ankle_angle_l'],
                                              )
            fig.savefig('results/motion_tracking_walking_inverse_'
                        'joint_moment_breakdown.png',
                        dpi=600)

        # report = osim.report.Report(model, mocoinverse_jr_solution_file)
        # report.generate()

        if plot_breakdown:
            fig = utilities.plot_joint_moment_breakdown(model,
                                                        sol_inverse_jointreaction,
                                              ['/jointset/hip_l/hip_flexion_l',
                                               '/jointset/hip_l/hip_adduction_l',
                                               '/jointset/hip_l/hip_rotation_l',
                                               '/jointset/walker_knee_l/knee_angle_l',
                                               '/jointset/ankle_l/ankle_angle_l'],
                                              )
            fig.savefig('results/motion_tracking_walking_inverse_'
                        'jointreaction_joint_moment_breakdown.png',
                        dpi=600)

        fig = plt.figure(figsize=(7.5, 3.5))
        gs = gridspec.GridSpec(3, 3)


        coords = [
            (
            f'/jointset/hip_{self.side}/hip_flexion_{self.side}', 'hip flexion',
            1.0),
            (f'/jointset/walker_knee_{self.side}/knee_angle_{self.side}',
             'knee flexion', 1.0),
            (f'/jointset/ankle_{self.side}/ankle_angle_{self.side}',
             'ankle plantarflexion', 1.0),
        ]
        from utilities import toarray
        # for ic, coord in enumerate(coords):
        #     ax = plt.subplot(gs[(3 * ic):(3 * (ic + 1)), 0])
        #
        #     y_cmc = coord[2] * np.rad2deg(
        #         toarray(sol_cmc.getDependentColumn(f'{coord[0]}/value')),)
        #     self.plot(ax, time_cmc, y_cmc, label='CMC', color='k',
        #               linewidth=3)
        #
        #     y_inverse = coord[2] * np.rad2deg(
        #         sol_inverse.getStateMat(f'{coord[0]}/value'))
        #     self.plot(ax, time_inverse, y_inverse, label='MocoInverse',
        #               linewidth=2)
        #
        #     ax.plot([0], [0], label='MocoInverse-knee', linewidth=2)
        #
        #     y_track = coord[2] * np.rad2deg(
        #         sol_track.getStateMat(f'{coord[0]}/value'))
        #     self.plot(ax, time_track, y_track, label='MocoTrack',
        #               linewidth=1)
        #
        #     ax.set_xlim(0, 100)
        #     if ic == 1:
        #         ax.legend(frameon=False, handlelength=1.9)
        #     if ic < len(coords) - 1:
        #         ax.set_xticklabels([])
        #     else:
        #         ax.set_xlabel('time (% gait cycle)')
        #         ax.get_xaxis().set_label_coords(0.5, 0)
        #         # The '0' would intersect with the y-axis, so remove it.
        #         ax.set_xticklabels(['', '20', '40', '60', '80', '100'])
        #     ax.set_ylabel(f'{coord[1]} (degrees)')
        #     ax.get_yaxis().set_label_coords(-0.15, 0.5)
        #
        #     ax.spines['bottom'].set_position('zero')
        #     utilities.publication_spines(ax)

        # TODO: Compare to EMG.
        muscles = [
            ((0, 0), 'glmax2', 'gluteus maximus', 'GMAX'),
            ((0, 1), 'psoas', 'psoas', ''),
            ((1, 0), 'semimem', 'semimembranosus', 'MH'),
            ((0, 2), 'recfem', 'rectus femoris', 'RF'),
            ((1, 1), 'bfsh', 'biceps femoris short head', 'BF'),
            ((1, 2), 'vaslat', 'vastus lateralis', 'VL'),
            ((2, 0), 'gasmed', 'medial gastrocnemius', 'GAS'),
            ((2, 1), 'soleus', 'soleus', 'SOL'),
            ((2, 2), 'tibant', 'tibialis anterior', 'TA'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[muscle[0][0], muscle[0][1]])
            activation_path = f'/forceset/{muscle[1]}_{self.side}/activation'
            cmc_activ = toarray(sol_cmc.getDependentColumn(activation_path))
            self.plot(ax, time_cmc,
                      cmc_activ,
                      linewidth=3,
                      label='CMC',
                      color='dimgray',
                    )
            self.plot(ax, time_track, sol_track.getStateMat(activation_path),
                          label='MocoTrack',
                          linewidth=2)
            self.plot(ax, time_inverse,
                      sol_inverse.getStateMat(activation_path),
                      label='MocoInverse',
                      linewidth=2)
            self.plot(ax, time_inverse_jointreaction,
                      sol_inverse_jointreaction.getStateMat(activation_path),
                      label='MocoInverse, knee',
                      linewidth=2)
            if len(muscle[3]) > 0:
                self.plot(ax, emg['time'], emg[muscle[3]] * np.max(cmc_activ),
                                      shift=False,
                                      fill=True,
                          color='lightgray')
            if muscle[0][0] == 0 and muscle[0][1] == 0:
                ax.legend(
                          frameon=False, handlelength=1.,
                    handletextpad=0.5,
                          ncol=2,
                    columnspacing=0.5,
                    loc='upper center',
                          # loc='center'
                )
            if muscle[0][0] == 1 and muscle[0][1] == 0:
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor='lightgray', label='EMG')],
                          frameon=False, handlelength=1.5,
                          handletextpad=0.5,
                          loc='upper center')
            ax.set_ylim(-0.05, 1)
            ax.set_xlim(0, 100)
            if muscle[0][0] < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            if muscle[0][1] == 0:
                ax.set_ylabel('activation')

            title = f'  {muscle[2]}'
            plt.text(0.5, 1.20, title,
                     horizontalalignment='center',
                     verticalalignment='top',
                     transform=ax.transAxes)
            ax.set_yticks([0, 1])

            utilities.publication_spines(ax)

        fig.tight_layout(h_pad=1)

        fig.savefig('figures/motion_tracking_walking.eps')
        fig.savefig('figures/motion_tracking_walking.pdf')
        fig.savefig('figures/motion_tracking_walking.png', dpi=600)

        res_track = self.calc_reserves(sol_track)
        column_labels = res_track.getColumnLabels()
        max_res_track = -np.inf
        for icol in range(res_track.getNumColumns()):
            column = utilities.toarray(
                res_track.getDependentColumnAtIndex(icol))
            max = np.max(np.abs(column))
            max_res_track = np.max([max_res_track, max])
            print(f'track max abs {column_labels[icol]}: {max}')
        with open('results/motion_tracking_walking_'
                  'track_max_reserve.txt', 'w') as f:
            f.write(f'{max_res_track:.2f}')

        res_inverse = self.calc_reserves(sol_inverse)
        column_labels = res_inverse.getColumnLabels()
        max_res_inverse = -np.inf
        for icol in range(res_inverse.getNumColumns()):
            column = utilities.toarray(
                res_inverse.getDependentColumnAtIndex(icol))
            max = np.max(np.abs(column))
            max_res_inverse = np.max([max_res_inverse, max])
            print(f'inverse max abs {column_labels[icol]}: {max}')
        with open('results/motion_tracking_walking_'
                  'inverse_max_reserve.txt', 'w') as f:
            f.write(f'{max_res_inverse:.1f}')

        res_inverse_jr = self.calc_reserves(sol_inverse_jointreaction)
        column_labels = res_inverse_jr.getColumnLabels()
        max_res_inverse_jr = -np.inf
        for icol in range(res_inverse_jr.getNumColumns()):
            column = utilities.toarray(
                res_inverse_jr.getDependentColumnAtIndex(icol))
            max = np.max(np.abs(column))
            max_res_inverse_jr = np.max([max_res_inverse_jr, max])
            print(f'inverse_jr max abs {column_labels[icol]}: {max}')
        with open('results/motion_tracking_walking_'
                  'inverse_jr_max_reserve.txt', 'w') as f:
            f.write(f'{max_res_inverse_jr:.3f}')

        states = sol_inverse.exportToStatesTrajectory(model)
        duration = sol_inverse.getFinalTime() - sol_inverse.getInitialTime()
        avg_speed = (model.calcMassCenterPosition(states[states.getSize() - 1])[0] -
                     model.calcMassCenterPosition(states[0])[0]) / duration
        print(f'Average speed: {avg_speed}')



        maxjr_inverse = self.calc_max_knee_reaction_force(sol_inverse)
        maxjr_inverse_jr = self.calc_max_knee_reaction_force(sol_inverse_jointreaction)
        print(f'Max joint reaction {maxjr_inverse} -> {maxjr_inverse_jr}')
        with open('results/motion_tracking_walking_'
                  'inverse_maxjr.txt', 'w') as f:
            f.write(f'{maxjr_inverse:.1f}')
        with open('results/motion_tracking_walking_'
                  'inverse_jr_maxjr.txt', 'w') as f:
            f.write(f'{maxjr_inverse_jr:.1f}')

class CrouchToStand(MocoPaperResult):
    def __init__(self):
        self.predict_solution_file = \
            'results/crouch_to_stand_predict_solution.sto'
        self.predict_assisted_solution_file = \
            'results/crouch_to_stand_predict_assisted_solution.sto'

    def create_study(self, model):
        moco = osim.MocoStudy()
        moco.set_write_solution("results/")
        solver = moco.initCasADiSolver()
        solver.set_dynamics_mode('implicit')
        solver.set_num_mesh_intervals(50)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_optim_constraint_tolerance(1e-3)
        solver.set_optim_finite_difference_scheme('forward')

        problem = moco.updProblem()
        problem.setModelCopy(model)
        problem.setTimeBounds(0, [0.1, 2])
        # problem.setTimeBounds(0, 1)
        problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value',
                          [-2, 0.5], -2, 0)
        problem.setStateInfo('/jointset/knee_r/knee_angle_r/value',
                          [-2, 0], -2, 0)
        problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value',
                          [-0.5, 0.7], -0.5, 0)
        problem.setStateInfoPattern('/jointset/.*/speed', [], 0, 0)
        # TODO: Minimize initial activation.

        # for muscle in model.getMuscles():
        #     if not muscle.get_ignore_activation_dynamics():
        #         muscle_path = muscle.getAbsolutePathString()
        #         problem.setStateInfo(muscle_path + '/activation', [0, 1], 0)
        #         problem.setControlInfo(muscle_path, [0, 1], 0)
        return moco

    def muscle_driven_model(self):
        model = osim.Model('resources/sitToStand_3dof9musc.osim')
        model.finalizeConnections()
        osim.DeGrooteFregly2016Muscle.replaceMuscles(model)
        for muscle in model.getMuscles():
            muscle.set_ignore_tendon_compliance(True)
            muscle.set_max_isometric_force(2 * muscle.get_max_isometric_force())
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            # dgf.set_active_force_width_scale(1.5)
            if muscle.getName() == 'soleus_r':
                dgf.set_ignore_passive_fiber_force(True)
        return model

    def predict(self):
        moco = self.create_study(self.muscle_driven_model())
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))
        problem.addGoal(osim.MocoFinalTimeGoal('time', 0.1))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        solution = moco.solve()
        solution.write(self.predict_solution_file)

        return solution

    def predict_assisted(self):
        model = self.muscle_driven_model()
        device = osim.SpringGeneralizedForce('knee_angle_r')
        device.setName('spring')
        device.setStiffness(50)
        device.setRestLength(0)
        device.setViscosity(0)
        model.addForce(device)

        moco = self.create_study(model)
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))

        problem.addParameter(
           osim.MocoParameter('stiffness', '/forceset/spring',
                              'stiffness', osim.MocoBounds(0, 100)))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        solver.set_parameters_require_initsystem(False)
        solution = moco.solve()
        # moco.visualize(solution)
        solution.write(self.predict_assisted_solution_file)

    def generate_results(self):
        self.predict()
        self.predict_assisted()

    def report_results(self):
        fig = plt.figure(figsize=(7.5, 3))
        values = [
            '/jointset/hip_r/hip_flexion_r/value',
            '/jointset/knee_r/knee_angle_r/value',
            '/jointset/ankle_r/ankle_angle_r/value',
        ]
        coord_names = {
            '/jointset/hip_r/hip_flexion_r/value': 'hip flexion',
            '/jointset/knee_r/knee_angle_r/value': 'knee flexion',
            '/jointset/ankle_r/ankle_angle_r/value': 'ankle dorsiflexion',
        }
        coord_signs = {
            '/jointset/hip_r/hip_flexion_r/value': -1.0,
            '/jointset/knee_r/knee_angle_r/value': -1.0,
            '/jointset/ankle_r/ankle_angle_r/value': -1.0,
        }

        muscles = [
            ((0, 3), 'glut_max2', 'gluteus maximus'),
            ((0, 2), 'psoas', 'iliopsoas'),
            # ('semimem', 'hamstrings'),
            ((1, 3), 'rect_fem', 'rectus femoris'),
            # ('bifemsh', 'biceps femoris short head'),
            ((1, 2), 'vas_int', 'vasti'),
            ((2, 2), 'med_gas', 'gastrocnemius'),
            # ('soleus', 'soleus'),
            ((2, 3), 'tib_ant', 'tibialis anterior'),
        ]
        # grid = plt.GridSpec(9, 2, hspace=0.7,
        #                     left=0.1, right=0.98, bottom=0.07, top=0.96,
        #                     )
        grid = gridspec.GridSpec(3, 4)

        ax = fig.add_subplot(grid[0, 0])
        import cv2
        # Convert BGR color ordering to RGB.
        image = cv2.imread('crouch_to_stand_visualization/'
                           'crouch_to_stand_visualization.png')[...,::-1]
        ax.imshow(image)
        plt.axis('off')

        coord_axes = []
        for ic, coordvalue in enumerate(values):
            ax = fig.add_subplot(grid[ic, 1])
            ax.set_ylabel('%s\n(deg.)' % coord_names[coordvalue])
            # ax.text(0.5, 1, '%s (degrees)' % coord_names[coordvalue],
            #         horizontalalignment='center',
            #         transform=ax.transAxes)
            ax.get_yaxis().set_label_coords(-0.20, 0.5)
            if ic == len(values) - 1:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            utilities.publication_spines(ax)
            ax.spines['bottom'].set_position('zero')
            coord_axes += [ax]
        muscle_axes = []
        for im, muscle in enumerate(muscles):
            ax = fig.add_subplot(grid[muscle[0][0], muscle[0][1]])
            # ax.set_title('%s activation' % muscle[1])
            title = f'  {muscle[2]}'
            plt.text(0.5, 0.9, title,
                     horizontalalignment='center',
                     transform=ax.transAxes
                     )
            ax.set_yticks([0, 1])
            ax.set_ylim([0, 1])
            if muscle[0][0] == 2:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            if muscle[0][1] == 2:
                ax.set_ylabel('activation')
            utilities.publication_spines(ax)
            muscle_axes += [ax]
        def plot_solution(sol, label, linestyle='-', color=None):
            time = sol.getTimeMat()
            for ic, coordvalue in enumerate(values):
                ax = coord_axes[ic]
                if ic == 1:
                    use_label = label
                else:
                    use_label = None
                if coordvalue in sol.getStateNames():
                    y = (coord_signs[coordvalue] * np.rad2deg(
                        sol.getStateMat(coordvalue)))
                    ax.plot(time, y, linestyle=linestyle, color=color,
                            label=use_label, clip_on=False)
                    ax.autoscale(enable=True, axis='x', tight=True)
                else:
                    if ic == 0:
                        ax.plot(0, 0, label=use_label)
            for im, muscle in enumerate(muscles):
                ax = muscle_axes[im]
                ax.plot(time,
                        sol.getStateMat(
                            '/forceset/%s_r/activation' % muscle[1]),
                        linestyle=linestyle, color=color, clip_on=False)
                ax.autoscale(enable=True, axis='x', tight=True)


        predict_solution = osim.MocoTrajectory(self.predict_solution_file)

        predict_assisted_solution = osim.MocoTrajectory(
            self.predict_assisted_solution_file)
        stiffness = predict_assisted_solution.getParameter('stiffness')

        # states = predict_assisted_solution.exportToStatesTrajectory(
        #     self.muscle_driven_model())

        print(f'Stiffness: {stiffness}')
        with open('results/crouch_to_stand_stiffness.txt', 'w') as f:
            f.write(f'{stiffness:.0f}')
        plot_solution(predict_solution, 'prediction', color='k')
        plot_solution(predict_assisted_solution, 'prediction, assisted')


        fig.tight_layout() # w_pad=0.2)

        coord_axes[1].legend(frameon=False, handlelength=1,
                             bbox_to_anchor=(-0.9, 0.5),
                             loc='center',
                             )

        # knee_angle = predict_assisted_solution.getStateMat(
        #     '/jointset/knee_r/knee_angle_r/value')
        # axright = coord_axes[1].twinx()
        # axright.plot(predict_assisted_solution.getTimeMat(),
        #              -stiffness * knee_angle)
        # axright.set_ylabel('knee extension moment (N-m)')
        # axright.set_yticks([0, 100, 200])
        # axright.spines['top'].set_visible(False)
        # axright.spines['bottom'].set_visible(False)

        fig.savefig('figures/crouch_to_stand.png', dpi=600)


        # fig = utilities.plot_joint_moment_breakdown(self.muscle_driven_model(),
        #                                   predict_solution,
        #                             ['/jointset/hip_r/hip_flexion_r',
        #                              '/jointset/knee_r/knee_angle_r',
        #                              '/jointset/ankle_r/ankle_angle_r'])
        # fig.savefig('figures/crouch_to_stand_joint_moment_contribution.png',
        #             dpi=600)
        # fig = utilities.plot_joint_moment_breakdown(self.muscle_driven_model(),
        #                             predict_assisted_solution,
        #                             ['/jointset/hip_r/hip_flexion_r',
        #                              '/jointset/knee_r/knee_angle_r',
        #                              '/jointset/ankle_r/ankle_angle_r'],
        #                             )
        # fig.savefig('figures/crouch_to_stand_assisted_'
        #             'joint_moment_contribution.png',
        #             dpi=600)

        sol_predict_table = osim.TimeSeriesTable(self.predict_solution_file)
        sol_predict_duration = sol_predict_table.getTableMetaDataString('solver_duration')
        sol_predict_duration = float(sol_predict_duration) / 60.0
        print('prediction duration ', sol_predict_duration)
        with open('results/'
                  'crouch_to_stand_predict_duration.txt', 'w') as f:
            f.write(f'{sol_predict_duration:.1f}')

        sol_predict_assisted_table = osim.TimeSeriesTable(self.predict_assisted_solution_file)
        sol_predict_assisted_duration = sol_predict_assisted_table.getTableMetaDataString('solver_duration')
        sol_predict_assisted_duration = float(sol_predict_assisted_duration) / 60.0
        print('prediction assisted duration ', sol_predict_assisted_duration)
        with open('results/'
                  'crouch_to_stand_predict_assisted_duration.txt', 'w') as f:
            f.write(f'{sol_predict_assisted_duration:.1f}')

if __name__ == "__main__":
    import argparse

    results = {
        'analytic': Analytic(),
        'suspended-mass': SuspendedMass(),
        'tracking-walking': MotionTrackingWalking(),
        # 'predicting-walking': MotionPredictionAndAssistanceWalking(),
        'crouch-to-stand': CrouchToStand(),
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
    parser.set_defaults(generate=True)
    args = parser.parse_args()

    for result_name, result_object in results.items():
        if args.results is None or result_name in args.results:
            if args.generate:
                print(f'Generating {result_name} results.')
                result_object.generate_results()
            print(f'Reporting {result_name} results.')
            result_object.report_results()

