import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pylab as pl

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities

suspended_mass_code = \
"""
study = MocoStudy();
problem = study.updProblem();
problem.addGoal(MocoControlGoal('effort'));
problem.addGoal(MocoFinalTimeGoal('time', 0.01));
problem.setModel(createSuspendedMassModel());
problem.setTimeBounds(0, [0.4, 0.8]);
problem.setStateInfo('/jointset/tx/tx/value', [-0.2, 0.2], -0.14, 0.14);
problem.setStateInfo('/jointset/ty/ty/value', [-0.4, 0], -0.2, 0.15);
problem.setStateInfoPattern('/jointset/.*/speed', [-15, 15], 0, 0);
problem.setStateInfoPattern('/forceset/.*/activation', [0, 1], 0):
problem.setControlInfoPattern('/forceset/.*', [0, 1]);
solution = study.solve()
"""

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
        problem.setStateInfoPattern("/jointset/.*/speed", [-15, 15], 0, 0)
        problem.setStateInfoPattern("/forceset/.*/activation", [0, 1], 0)
        problem.setControlInfoPattern("/forceset/.*", [0, 1])
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

    def generate_results(self, root_dir, args):
        predict_solution, time_stepping = self.predict(True)
        predict_solution.write(
            os.path.join(root_dir, 'results/suspended_mass_prediction_solution.sto'))
        time_stepping.write(
            os.path.join(root_dir, 'results/suspended_mass_time_stepping.sto'))
        track_solution = self.track(predict_solution)
        track_solution.write(
            os.path.join(root_dir, 'results/suspended_mass_track_solution.sto'))
        track_solution_p = self.track(predict_solution, 4)
        track_solution_p.write(
            os.path.join(root_dir, 'results/suspended_mass_track_p_solution.sto'))

    def report_results(self, root_dir, args):
        predict_solution = osim.MocoTrajectory(
            os.path.join(root_dir, 'results/suspended_mass_prediction_solution.sto'))
        time_stepping = osim.MocoTrajectory(
            os.path.join(root_dir, 'results/suspended_mass_time_stepping.sto'))
        track_solution = osim.MocoTrajectory(
            os.path.join(root_dir, 'results/suspended_mass_track_solution.sto'))
        track_p_solution = osim.MocoTrajectory(
            os.path.join(root_dir, 'results/suspended_mass_track_p_solution.sto'))

        peak_left = np.max(predict_solution.getStateMat('/forceset/left/activation'))
        peak_middle = np.max(predict_solution.getStateMat('/forceset/middle/activation'))
        peak_right = np.max(predict_solution.getStateMat('/forceset/right/activation'))
        peak_predicted_activation = np.max([peak_left, peak_middle, peak_right])

        time_stepping_rms = time_stepping.compareContinuousVariablesRMSPattern(
            predict_solution,
            'states', '/jointset.*value$')
        print(f'time-stepping rms: {time_stepping_rms}')
        with open(os.path.join(root_dir, 'results/suspended_mass_'
                  'time_stepping_coord_rms.txt'), 'w') as f:
            f.write(f'{time_stepping_rms:.4f}')
        distance = np.sqrt(
            (self.xfinal - self.xinit) ** 2 + (self.yfinal - self.yinit) ** 2)
        norm_rms = 100.0 * time_stepping_rms / distance
        print(f'time-stepping rms as a percent of distance: {norm_rms}')
        with open(os.path.join(root_dir, 'results/suspended_mass_'
                                         'time_stepping_coord_norm_rms.txt'), 'w') as f:
            f.write(f'{norm_rms:.1f}')

        track_rms = track_solution.compareContinuousVariablesRMSPattern(
            predict_solution, 'states', '/forceset.*activation$')
        track_rms_pcent = 100.0 * track_rms / peak_predicted_activation
        print(f'track rms percentage of peak activation: {track_rms_pcent}')
        with open(os.path.join(root_dir, 'results/suspended_mass_'
                  'track_activation_rms.txt'), 'w') as f:
            f.write(f'{track_rms_pcent:.2f}')

        track_p_rms = track_p_solution.compareContinuousVariablesRMSPattern(
            predict_solution, 'states', '/forceset.*activation$')
        track_p_rms_pcent = 100.0 * track_p_rms / peak_predicted_activation
        print(f'track p=4 rms: {track_p_rms_pcent}')
        with open(os.path.join(root_dir, 'results/suspended_mass_'
                  'track_p_activation_rms.txt'), 'w') as f:
            f.write(f'{track_p_rms_pcent:.0f}')

        fig = plt.figure(figsize=(5.2, 4.2))
        grid = gridspec.GridSpec(4, 4,
                                 height_ratios=[4, 1, 1, 1])

        ax = fig.add_subplot(grid[0, :])
        ax.text(0.2, 0.5, suspended_mass_code, fontsize=8,
                verticalalignment='center',
                # fontdict={'fontfamily': 'monospace'},
                transform=ax.transAxes)
        utilities.publication_spines(ax)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(grid[1:, 0:2])

        initial_alpha = 0.1
        ax.plot([-self.width, self.xinit],
                [0, self.yinit], color='tab:red', alpha=initial_alpha,
                linewidth=3)
        ax.plot([0, self.xinit],
                [0, self.yinit], color='tab:red', alpha=initial_alpha,
                linewidth=3)
        ax.plot([self.width, self.xinit],
                [0, self.yinit], color='tab:red', alpha=initial_alpha,
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
        ax.plot([-1.1 * self.width, 1.1 * self.width], [0.004, 0.004],
                color='k',
                linewidth=2.5)

        ax.plot([-self.width, 0, self.width],
                np.zeros(3),
                linestyle='',
                markersize=4,
                marker='o', color='black')

        ax.annotate('', xy=(0, -0.85 * self.width), xycoords='data',
                    xytext=(0, -0.7 * self.width),
                    arrowprops=dict(width=0.2, headwidth=2, headlength=1.5,
                                    facecolor='black'),
                    )
        ax.text(0, -0.78 * self.width, ' g')

        gray = (0.7, 0.7, 0.7)
        track_linewidth = 1.25
        stepping_linewidth = 3.25

        a = ax.plot(predict_solution.getStateMat('/jointset/tx/tx/value'),
                    predict_solution.getStateMat('/jointset/ty/ty/value'),
                    color=gray, linewidth=6,
                    label='prediction')
        b = ax.plot(time_stepping.getStateMat('/jointset/tx/tx/value'),
                    time_stepping.getStateMat('/jointset/ty/ty/value'),
                    linewidth=stepping_linewidth,
                    label='time-stepping',
                    )
        c = ax.plot(track_solution.getStateMat('/jointset/tx/tx/value'),
                    track_solution.getStateMat('/jointset/ty/ty/value'),
                    linewidth=track_linewidth,
                    label='MocoTrack')
        # d = ax.plot(track_p_solution.getStateMat('/jointset/tx/tx/value'),
        #             track_p_solution.getStateMat('/jointset/ty/ty/value'),
        #             linestyle='-', linewidth=1,
        #             label='MocoTrack, $x^4$')
        plt.annotate('initial', (self.xinit, self.yinit),
                     xytext=(self.xinit - 0.02, self.yinit - 0.04),
                     color='tab:red',
                     alpha=0.5)
        plt.annotate('final', (self.xfinal, self.yfinal),
                     xytext=(self.xfinal + 0.008, self.yfinal - 0.04),
                     color='tab:red',
                     )
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
        ax.set_title('trajectory of point mass', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'datalim')
        ax.legend(frameon=False,
                  handlelength=1.9,
                  loc='upper center')
        utilities.publication_spines(ax)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


        ax = fig.add_subplot(grid[1, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/left/activation'),
                color=gray, linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/left/activation'),
                linewidth=stepping_linewidth, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/left/activation'),
                linewidth=track_linewidth, clip_on=False)
        # ax.plot(track_p_solution.getTimeMat(),
        #         track_p_solution.getStateMat('/forceset/left/activation'),
        #         linestyle='-', linewidth=1, clip_on=False)
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

        ax = fig.add_subplot(grid[2, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/middle/activation'),
                color=gray, linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/middle/activation'),
                linewidth=stepping_linewidth, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/middle/activation'),
                linewidth=track_linewidth, clip_on=False)
        # ax.plot(track_p_solution.getTimeMat(),
        #         track_p_solution.getStateMat('/forceset/middle/activation'),
        #         linewidth=1, clip_on=False)
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

        ax = fig.add_subplot(grid[3, 2:4])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/right/activation'),
                color=gray, linewidth=6, clip_on=False)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/right/activation'),
                linewidth=stepping_linewidth, clip_on=False)
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/right/activation'),
                linewidth=track_linewidth, clip_on=False)
        # ax.plot(track_p_solution.getTimeMat(),
        #         track_p_solution.getStateMat('/forceset/right/activation'),
        #         linewidth=1, clip_on=False)
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

        pl.savefig(os.path.join(root_dir, 'figures/suspended_mass.png'),
                   dpi=600)
