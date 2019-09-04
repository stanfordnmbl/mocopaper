from abc import ABC, abstractmethod
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.lines import Line2D
import pylab as pl

import opensim as osim

# TODO: show icons of different muscles next to the activation plots.
# TODO: report reserve and residual forces.
# TODO: create a docker container for these results and generating the preprint.
# TODO fix shift
# TODO: report difference in knee joint loading between MocoInverse and
#  MocoInverse-knee.
# TODO: Put a gap in the plots for walking at 60pgc.
# TODO: Add a periodicity cost to walking.
# TODO: Use a lighter gray color for CMC. Generally fix the overlap of colors.
# TODO: Use MocoTrack in verification section.
# TODO: Verification: add a prediction for p=5.
# TODO: Add analytic problem to this file.
# TODO: crouch to stand: plot assistive torque?
# TODO: add diagram of motion.

mpl.rcParams.update({'font.size': 8,
                     'axes.titlesize': 8,
                     'axes.labelsize': 8,
                     'font.sans-serif': ['Arial'],
                     'image.cmap': 'Tab10'})

import utilities

class MocoPaperResult(ABC):
    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass


class SuspendedMass(MocoPaperResult):
    width = 0.2

    def __init__(self):
        pass
    def build_model(self):
        model = osim.ModelFactory.createPlanarPointMass()
        body = model.updBodySet().get("body")
        model.updForceSet().clearAndDestroy()
        model.finalizeFromProperties()

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
        problem.setTimeBounds(0, 0.5)
        problem.setStateInfo("/jointset/tx/tx/value", [-0.03, 0.03], -0.03,
                             0.03)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * self.width, 0],
                             -self.width,
                             -self.width + 0.05)
        problem.setStateInfo("/jointset/tx/tx/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/jointset/ty/ty/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.setControlInfo("/forceset/left", [0, 1])
        problem.setControlInfo("/forceset/middle", [0, 1])
        problem.setControlInfo("/forceset/right", [0, 1])

        solver = study.initCasADiSolver()
        solver.set_num_mesh_points(101)

        return study

    def predict(self):

        study = self.create_study()
        problem = study.updProblem()

        problem.addGoal(osim.MocoControlGoal())
        solution = study.solve()

        # study.visualize(solution)

        time_stepping = \
            osim.simulateIterateWithTimeStepping(solution, self.build_model())

        return solution, time_stepping

    def track(self, prediction_solution, exponent=2):

        study = self.create_study()
        problem = study.updProblem()

        tracking = osim.MocoStateTrackingGoal("tracking", 10.0)
        tracking.setReference(
            osim.TableProcessor(prediction_solution.exportToStatesTable()))
        tracking.setPattern('.*value$')
        tracking.setAllowUnusedReferences(True)
        problem.addGoal(tracking)
        effort = osim.MocoControlGoal("effort")
        effort.setExponent(exponent)
        problem.addGoal(effort)

        solution = study.solve()

        return solution

    def generate_results(self):
        predict_solution, time_stepping = self.predict()
        predict_solution.write('results/suspended_mass_prediction_solution.sto')
        time_stepping.write('results/suspended_mass_time_stepping.sto')
        track_solution = self.track(predict_solution)
        track_solution.write('results/suspended_mass_track_solution.sto')
        track_solution_p = self.track(predict_solution, 5)
        track_solution_p.write('results/suspended_mass_track_p_solution.sto')

    def report_results(self):
        pl.figure()
        fig = plt.figure(figsize=(5.5, 3))
        grid = gridspec.GridSpec(3, 2)
        predict_solution = osim.MocoTrajectory(
            'results/suspended_mass_prediction_solution.sto')
        time_stepping = osim.MocoTrajectory(
            'results/suspended_mass_time_stepping.sto')
        track_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_solution.sto')
        track_p_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_p_solution.sto')

        ax = fig.add_subplot(grid[0:3, 0])
        ax.plot(predict_solution.getStateMat('/jointset/tx/tx/value'),
                predict_solution.getStateMat('/jointset/ty/ty/value'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getStateMat('/jointset/tx/tx/value'),
                time_stepping.getStateMat('/jointset/ty/ty/value'),
                linestyle='--')
        ax.plot(track_solution.getStateMat('/jointset/tx/tx/value'),
                track_solution.getStateMat('/jointset/ty/ty/value'),
                linestyle=':')
        ax.plot(track_p_solution.getStateMat('/jointset/tx/tx/value'),
                track_p_solution.getStateMat('/jointset/ty/ty/value'),
                linestyle=':')
        plt.annotate('start', (-0.03, -self.width),
                     xytext=(-0.03, -self.width + 0.002))
        plt.annotate('end', (0.03, -self.width + 0.05),
                     xytext=(0.03 - 0.007, -self.width + 0.05))
        scalebar = AnchoredSizeBar(ax.transData, 0.01, label='1 cm',
                                   # loc=(0, 0.3),
                                   loc='center left',
                                   pad=0.5, frameon=False)
        ax.add_artist(scalebar)
        ax.set_title('trajectory of point mass')
        # ax.set_ylabel('y (m)')
        # ax.set_xlabel('x (m)')
        # ax.set_yticks([-0.20, -0.18, -0.16])
        # ax.set_xticks([-0.02, 0, 0.02])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.legend(['prediction p=2',
                   'time stepping',
                   'tracking p=2',
                   'tracking p=5'],
                  frameon=False,
                  handlelength=1.9)
        utilities.publication_spines(ax)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax = fig.add_subplot(grid[0, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/left/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/left/activation'),
                linestyle='--')
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/left/activation'),
                linestyle=':')
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/left/activation'),
                linestyle=':')
        ax.set_title('left activation')
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        utilities.publication_spines(ax)

        ax = fig.add_subplot(grid[1, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/middle/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/middle/activation'),
                linestyle='--')
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/middle/activation'),
                linestyle=':')
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/middle/activation'),
                linestyle=':')
        ax.set_title('middle activation')
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        utilities.publication_spines(ax)

        ax = fig.add_subplot(grid[2, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/right/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/right/activation'),
                linestyle='--',)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/right/activation'),
                linestyle=':',)
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/right/activation'),
                linestyle=':',)
        ax.set_title('right activation')
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel('time (s)')
        utilities.publication_spines(ax)

        fig.tight_layout()

        pl.savefig('figures/suspended_mass.png', dpi=600)
        pl.savefig('figures/suspended_mass.eps')
        pl.savefig('figures/suspended_mass.pdf')

    # TODO surround the point with muscles and maximize distance traveled.


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

    def shift(self, time, y):
        return utilities.shift_data_to_cycle(self.initial_time, self.final_time,
                                   self.footstrike, time, y, cut_off=False)

    def create_model_processor(self):
        modelProcessor = osim.ModelProcessor(
            # "resources/ArnoldSubject02Walk3/subject02_armless.osim")
            "resources/Rajagopal2016/subject_walk_armless.osim")
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        # modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
        modelProcessor.append(osim.ModOpAddReserves(1))
        modelProcessor.process().printToXML("subject_armless_for_cmc.osim")
        # ext_loads_xml = "resources/ArnoldSubject02Walk3/external_loads.xml"
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
        #  - play with weights between tracking and effort.
        #  - report duration to solve the problem.
        #  - figure could contain still frames of the model throughout the motion.
        #  - make sure residuals and reserves are small (generate_report).
        #  - try using Millard muscle.
        #  - plot joint moment breakdown.

        coordinates = osim.TableProcessor(
            "resources/Rajagopal2016/coordinates.sto")
        coordinates.append(osim.TabOpLowPassFilter(6))
        track.setStatesReference(coordinates)
        track.set_states_global_tracking_weight(0.1)

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
        # cmc.run()

        # TODO: why is recfem used instead of vaslat? recfem counters the hip
        # extension moment in early stance.

        # TODO compare to MocoInverse.
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.01)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_tolerance(1e-3)
        # 2 minutes
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

        # solution_reaction = study.solve()
        # solution_reaction.write(self.mocoinverse_jointreaction_solution_file)

        # TODO: Minimize joint reaction load!
    def plot(self, ax, time, y, *args, **kwargs):
        shifted_time, shifted_y = self.shift(time, y)
        # TODO is this correct?
        duration = self.final_time - self.initial_time
        ax.plot(100.0 * shifted_time / duration, shifted_y, *args,
                clip_on=False, **kwargs)

    def report_results(self):
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

        report = osim.report.Report(model, self.mocotrack_solution_file)
        report.generate()

        report = osim.report.Report(model, self.mocoinverse_solution_file)
        report.generate()

        # TODO: slight shift in CMC solution might be due to how we treat
        # percent gait cycle and the fact that CMC is missing 0.02 seconds.
        sol_cmc = osim.TimeSeriesTable('results/motion_tracking_walking_cmc_results/'
                                       'motion_tracking_walking_cmc_states.sto')
        # TODO uh oh overwriting the MocoInverse solution.
        # sol_inverse.setStatesTrajectory(sol_cmc)
        time_cmc = np.array(sol_cmc.getIndependentColumn())

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

        report = osim.report.Report(model, mocoinverse_jr_solution_file)
        report.generate()

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

        fig = plt.figure(figsize=(5.5, 5.5))
        gs = gridspec.GridSpec(9, 2)


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
        for ic, coord in enumerate(coords):
            ax = plt.subplot(gs[(3 * ic):(3 * (ic + 1)), 0])

            y_cmc = coord[2] * np.rad2deg(
                toarray(sol_cmc.getDependentColumn(f'{coord[0]}/value')),)
            self.plot(ax, time_cmc, y_cmc, label='CMC', color='gray')

            y_track = coord[2] * np.rad2deg(
                sol_track.getStateMat(f'{coord[0]}/value'))
            self.plot(ax, time_track, y_track, label='MocoTrack',
                      color='k',
                      linestyle='--')

            y_inverse = coord[2] * np.rad2deg(
                sol_inverse.getStateMat(f'{coord[0]}/value'))
            self.plot(ax, time_inverse, y_inverse, label='MocoInverse',
                      linestyle='--')

            # y_inverse_jr = coord[2] * np.rad2deg(
            #     sol_inverse_jointreaction.getStateMat(f'{coord[0]}/value'))
            # self.plot(ax, time_inverse_jointreaction, y_inverse_jr,
            #           label='MocoInverse-knee',
            #           linestyle='--')
            ax.plot([0], [0], label='MocoInverse-knee', linestyle='--')

            ax.set_xlim(0, 100)
            if ic == 1:
                ax.legend(frameon=False, handlelength=1.9)
            if ic < len(coords) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
                ax.get_xaxis().set_label_coords(0.5, 0)
                # The '0' would intersect with the y-axis, so remove it.
                ax.set_xticklabels(['', '20', '40', '60', '80', '100'])
            ax.set_ylabel(f'{coord[1]} (degrees)')
            ax.get_yaxis().set_label_coords(-0.15, 0.5)

            ax.spines['bottom'].set_position('zero')
            utilities.publication_spines(ax)

        # TODO: Compare to EMG.
        muscles = [
            ('glmax2', 'gluteus maximus'),
            ('psoas', 'psoas'),
            ('semimem', 'semimembranosus'),
            ('recfem', 'rectus femoris'),
            ('bfsh', 'biceps femoris short head'),
            ('vaslat', 'vastus lateralis'),
            ('gasmed', 'medial gastrocnemius'),
            ('soleus', 'soleus'),
            ('tibant', 'tibialis anterior'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[im, 1])
            activation_path = f'/forceset/{muscle[0]}_{self.side}/activation'
            self.plot(ax, time_cmc,
                      toarray(sol_cmc.getDependentColumn(activation_path)),
                    color='gray')
            self.plot(ax, time_track, sol_track.getStateMat(activation_path),
                    label='Track',
                    color='k', linestyle='--')
            self.plot(ax, time_inverse,
                      sol_inverse.getStateMat(activation_path),
                      label='Inverse',
                      linestyle='--')
            self.plot(ax, time_inverse_jointreaction,
                      sol_inverse_jointreaction.getStateMat(activation_path),
                      label='Inverse JR',
                      linestyle='--')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            if im < len(muscles) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')

            title = f'  {muscle[1]}'
            if im == 0:
                title += ' activation'
            plt.text(0, 1, title,
                     horizontalalignment='left',
                     verticalalignment='top')
            ax.set_yticks([0, 1])

            utilities.publication_spines(ax)

        fig.tight_layout(h_pad=0.42)

        fig.savefig('figures/motion_tracking_walking.eps')
        fig.savefig('figures/motion_tracking_walking.pdf')
        fig.savefig('figures/motion_tracking_walking.png', dpi=600)

class CrouchToStand(MocoPaperResult):
    def __init__(self):
        self.predict_solution_file = 'predict_solution.sto'
        self.predict_assisted_solution_file = 'predict_assisted_solution.sto'
        self.track_solution_file = 'track_solution.sto'
        self.inverse_solution_file = 'inverse_solution.sto'
        self.inverse_assisted_solution_file = 'inverse_assisted_solution.sto'

    def create_study(self, model):
        moco = osim.MocoStudy()
        moco.set_write_solution("results/")
        solver = moco.initCasADiSolver()
        solver.set_dynamics_mode('implicit')
        solver.set_num_mesh_points(50)
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
        moco.visualize(solution)
        solution.write(self.predict_assisted_solution_file)

    def generate_results(self):
        self.predict()
        self.predict_assisted()

    def report_results(self):
        fig = plt.figure(figsize=(5.5, 4.5))
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
            ('glut_max2', 'gluteus maximus'),
            ('psoas', 'iliopsoas'),
            # ('semimem', 'hamstrings'),
            ('rect_fem', 'rectus femoris'),
            # ('bifemsh', 'biceps femoris short head'),
            ('vas_int', 'vasti'),
            ('med_gas', 'gastrocnemius'),
            # ('soleus', 'soleus'),
            ('tib_ant', 'tibialis anterior'),
        ]
        # grid = plt.GridSpec(9, 2, hspace=0.7,
        #                     left=0.1, right=0.98, bottom=0.07, top=0.96,
        #                     )
        grid = gridspec.GridSpec(6, 2)
        coord_axes = []
        for ic, coordvalue in enumerate(values):
            ax = fig.add_subplot(grid[2 * ic: 2 * (ic + 1), 0])
            ax.set_ylabel('%s (degrees)' % coord_names[coordvalue])
            ax.get_yaxis().set_label_coords(-0.15, 0.5)
            if ic == len(values) - 1:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            utilities.publication_spines(ax)
            ax.spines['bottom'].set_position('zero')
            coord_axes += [ax]
        muscle_axes = []
        for im, muscle in enumerate(muscles):
            ax = fig.add_subplot(grid[im, 1])
            # ax.set_title('%s activation' % muscle[1])
            title = f'  {muscle[1]}'
            if im == 0:
                title += ' activation'
            plt.text(0.5, 0.9, title,
                     horizontalalignment='center',
                     transform=ax.transAxes
                     )
            ax.set_yticks([0, 1])
            ax.set_ylim([0, 1])
            if im == len(muscles) - 1:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            utilities.publication_spines(ax)
            muscle_axes += [ax]
        def plot_solution(sol, label, linestyle='-', color=None):
            time = sol.getTimeMat()
            for ic, coordvalue in enumerate(values):
                ax = coord_axes[ic]
                if ic == 0:
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
                            '/forceset/%s_r/activation' % muscle[0]),
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
            f.write(f'{stiffness}')
        plot_solution(predict_solution, 'prediction', '-', 'k')
        plot_solution(predict_assisted_solution, 'prediction with assistance',
                      linestyle='--')

        coord_axes[0].legend(frameon=False, handlelength=1.9)

        # knee_angle = predict_assisted_solution.getStateMat(
        #     '/jointset/knee_r/knee_angle_r/value')
        # axright = coord_axes[1].twinx()
        # axright.plot(predict_assisted_solution.getTimeMat(),
        #              -stiffness * knee_angle)
        # axright.set_ylabel('knee extension moment (N-m)')
        # axright.set_yticks([0, 100, 200])
        # axright.spines['top'].set_visible(False)
        # axright.spines['bottom'].set_visible(False)

        fig.tight_layout(h_pad=0.4)
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

if __name__ == "__main__":
    import argparse

    results = {
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

