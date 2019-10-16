import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities

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
        solver.set_multibody_dynamics_mode('implicit')
        solver.set_minimize_implicit_multibody_accelerations(True)
        solver.set_implicit_multibody_accelerations_weight(0.001)
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        solver.set_implicit_auxiliary_derivatives_weight(0.001)
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
            # muscle.set_ignore_tendon_compliance(True)
            muscle.set_max_isometric_force(2 * muscle.get_max_isometric_force())
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            # dgf.set_active_force_width_scale(1.5)
            dgf.set_tendon_compliance_dynamics_mode('implicit')
            if muscle.getName() == 'soleus_r':
                dgf.set_ignore_passive_fiber_force(True)
        model.printToXML("resources/sitToStand_3dof9musc_dgf.osim")
        return model

    def predict(self):
        model = self.muscle_driven_model()
        moco = self.create_study(model)
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))
        problem.addGoal(osim.MocoFinalTimeGoal('time', 0.5))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        guess = solver.createGuess()

        N = guess.getNumTimes()
        for muscle in model.getMuscles():
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            if not dgf.get_ignore_tendon_compliance():
                guess.setState(
                    '%s/normalized_tendon_force' % muscle.getAbsolutePathString(),
                    osim.createVectorLinspace(N, 0.1, 0.1))
        solver.setGuess(guess)

        solution = moco.solve()
        solution.write(self.predict_solution_file)

        return solution

    def assisted_model(self):
        model = self.muscle_driven_model()
        device = osim.SpringGeneralizedForce('knee_angle_r')
        device.setName('spring')
        device.setStiffness(50)
        device.setRestLength(0)
        device.setViscosity(0)
        model.addForce(device)
        return model

    def predict_assisted(self):
        model = self.assisted_model()

        moco = self.create_study(model)
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))
        problem.addGoal(osim.MocoFinalTimeGoal('time', 0.5))

        problem.addParameter(
            osim.MocoParameter('stiffness', '/forceset/spring',
                               'stiffness', osim.MocoBounds(0, 100)))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        guess = solver.createGuess()

        N = guess.getNumTimes()
        for muscle in model.getMuscles():
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            if not dgf.get_ignore_tendon_compliance():
                guess.setState(
                    '%s/normalized_tendon_force' % muscle.getAbsolutePathString(),
                    osim.createVectorLinspace(N, 0.1, 0.1))
        solver.setGuess(guess)

        solver.set_parameters_require_initsystem(False)
        solution = moco.solve()
        # moco.visualize(solution)
        solution.write(self.predict_assisted_solution_file)

    def generate_results(self, args):
        self.predict()
        self.predict_assisted()

    def report_results(self, args):
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
            # ((0, 3), 'glut_max2', 'gluteus maximus'),
            ((0, 2), 'psoas', 'iliopsoas'),
            # ('semimem', 'hamstrings'),
            ((1, 2), 'rect_fem', 'rectus femoris'),
            # ('bifemsh', 'biceps femoris short head'),
            ((2, 2), 'vas_int', 'vasti'),
            # ((2, 2), 'med_gas', 'gastrocnemius'),
            # ('soleus', 'soleus'),
            ((3, 2), 'tib_ant', 'tibialis anterior'),
        ]
        # grid = plt.GridSpec(9, 2, hspace=0.7,
        #                     left=0.1, right=0.98, bottom=0.07, top=0.96,
        #                     )
        grid = gridspec.GridSpec(4, 3)

        ax = fig.add_subplot(grid[0:4, 0:2])
        import cv2
        # Convert BGR color ordering to RGB.
        image = cv2.imread('crouch_to_stand_visualization/'
                           'crouch_to_stand_visualization.png')[...,::-1]
        ax.imshow(image)
        plt.axis('off')

        # coord_axes = []
        # for ic, coordvalue in enumerate(values):
        #     ax = fig.add_subplot(grid[ic, 1])
        #     ax.set_ylabel('%s\n(deg.)' % coord_names[coordvalue])
        #     # ax.text(0.5, 1, '%s (degrees)' % coord_names[coordvalue],
        #     #         horizontalalignment='center',
        #     #         transform=ax.transAxes)
        #     ax.get_yaxis().set_label_coords(-0.20, 0.5)
        #     if ic == len(values) - 1:
        #         ax.set_xlabel('time (s)')
        #     else:
        #         ax.set_xticklabels([])
        #     utilities.publication_spines(ax)
        #     ax.spines['bottom'].set_position('zero')
        #     coord_axes += [ax]
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
            if muscle[0][0] == 3:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            if muscle[0][1] == 2:
                ax.set_ylabel('activation')
            utilities.publication_spines(ax)
            muscle_axes += [ax]
        def plot_solution(sol, label, linestyle='-', color=None):
            time = sol.getTimeMat()
            # for ic, coordvalue in enumerate(values):
            #     ax = coord_axes[ic]
            #     if ic == 1:
            #         use_label = label
            #     else:
            #         use_label = None
            #     if coordvalue in sol.getStateNames():
            #         y = (coord_signs[coordvalue] * np.rad2deg(
            #             sol.getStateMat(coordvalue)))
            #         ax.plot(time, y, linestyle=linestyle, color=color,
            #                 label=use_label, clip_on=False)
            #         ax.autoscale(enable=True, axis='x', tight=True)
            #     else:
            #         if ic == 0:
            #             ax.plot(0, 0, label=use_label)
            for im, muscle in enumerate(muscles):
                ax = muscle_axes[im]
                ax.plot(time,
                        sol.getStateMat(
                            '/forceset/%s_r/activation' % muscle[1]),
                        linestyle=linestyle, color=color, clip_on=False,
                        label=label)
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
        plot_solution(predict_solution, 'unassisted', color='k')
        plot_solution(predict_assisted_solution, 'assisted')


        fig.tight_layout() # w_pad=0.2)

        muscle_axes[0].legend(frameon=False, handlelength=1,
                             bbox_to_anchor=(-1.0, 0.5),
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

        model = self.muscle_driven_model()
        report = osim.report.Report(self.muscle_driven_model(),
                                    self.predict_solution_file)
        report.generate()

        # table = osim.analyze(model,
        #                      osim.MocoTrajectory(self.predict_solution_file),
        #              ['.*normalized_fiber_length'])
        # osim.STOFileAdapter.write(table,
        #                           'crouch_to_stand_norm_fiber_length.sto')

        report = osim.report.Report(self.assisted_model(),
                                    self.predict_assisted_solution_file)
        report.generate()
