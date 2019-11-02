import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities


class SquatToStand(MocoPaperResult):
    def __init__(self):
        self.predict_solution_file = \
            'results/squat_to_stand_predict_solution.sto'
        self.predict_assisted_solution_file = \
            'results/squat_to_stand_predict_assisted_solution.sto'

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
        # solver.set_output_interval(10)

        problem = moco.updProblem()
        problem.setModelCopy(model)
        problem.setTimeBounds(0, [0.1, 2])
        # Initial squat pose is based on
        # https://simtk.org/projects/aredsimulation
        # Simulation Files/ISS_ARED_Initial_States.sto; t = 1.0 seconds;
        # Given that our model is planar, we adjust the squat so that the
        # system's center of mass is over the feet.
        # This avoid excessive tib ant activity simply to prevent the model
        # from falling backwards, which is unrealistic.
        squat_lumbar = np.deg2rad(-22)
        problem.setStateInfo('/jointset/back/lumbar_extension/value',
                             [1.5 * squat_lumbar, 0.5], squat_lumbar, 0)
        squat_hip = np.deg2rad(-98)
        problem.setStateInfo('/jointset/hip_r/hip_extension_r/value',
                             [1.5 * squat_hip, 0.5], squat_hip, 0)
        squat_knee = np.deg2rad(-104)
        problem.setStateInfo('/jointset/knee_r/knee_extension_r/value',
                             [1.5 * squat_knee, 0], squat_knee, 0)
        squat_ankle = np.deg2rad(-30)
        problem.setStateInfo('/jointset/ankle_r/ankle_plantarflexion_r/value',
                             [1.5 * squat_ankle, 0.5], squat_ankle, 0)
        problem.setStateInfoPattern('/jointset/.*/speed', [], 0, 0)

        return moco

    def muscle_driven_model(self):
        model = osim.Model('resources/squat_to_stand_4dof9musc.osim')
        model.finalizeConnections()
        osim.DeGrooteFregly2016Muscle.replaceMuscles(model)
        for muscle in model.getMuscles():
            # Missing a leg: double the forces.
            muscle.set_max_isometric_force(
                2.0 * muscle.get_max_isometric_force())
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            dgf.set_ignore_passive_fiber_force(True)
            dgf.set_tendon_compliance_dynamics_mode('implicit')
            # if muscle.getName() == 'soleus_r':
            #     dgf.set_ignore_passive_fiber_force(True)
        model.printToXML("resources/squat_to_stand_4dof9musc_dgf.osim")
        return model

    def predict(self):
        model = self.muscle_driven_model()
        moco = self.create_study(model)
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))
        problem.addGoal(osim.MocoFinalTimeGoal('time'))

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
            guess.setState(
                '%s/activation' % muscle.getAbsolutePathString(),
                osim.createVectorLinspace(N, 0.05, 0.05))
            guess.setControl(muscle.getAbsolutePathString(),
                osim.createVectorLinspace(N, 0.05, 0.05))
        solver.setGuess(guess)

        solution = moco.solve()
        solution.write(self.predict_solution_file)
        # moco.visualize(solution)

        return solution

    def assisted_model(self):
        model = self.muscle_driven_model()
        device = osim.SpringGeneralizedForce('knee_extension_r')
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
        problem.addGoal(osim.MocoFinalTimeGoal('time'))

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
            guess.setState(
                '%s/activation' % muscle.getAbsolutePathString(),
                osim.createVectorLinspace(N, 0.05, 0.05))
            guess.setControl(muscle.getAbsolutePathString(),
                             osim.createVectorLinspace(N, 0.05, 0.05))
        solver.setGuess(guess)

        solver.set_parameters_require_initsystem(False)
        solution = moco.solve()
        # moco.visualize(solution)
        solution.write(self.predict_assisted_solution_file)

    def parse_args(self, args):
        self.unassisted = False
        self.assisted = False
        if len(args) == 0:
            self.unassisted = True
            self.assisted = True
            return
        print('Received arguments {}'.format(args))
        if 'unassisted' in args:
            self.unassisted = True
        if 'assisted' in args:
            self.assisted = True

    def generate_results(self, args):
        self.parse_args(args)
        if self.unassisted:
            self.predict()
        if self.assisted:
            self.predict_assisted()

    def plot_joint_moment_breakdown(self, model, moco_traj,
                                    coord_paths, muscle_paths=None,
                                    coordact_paths=[], knee_stiffness=None):
        model.initSystem()

        num_coords = len(coord_paths)

        coord_names = {
            '/jointset/hip_r/hip_extension_r/value': 'hip extension',
            '/jointset/knee_r/knee_extension_r/value': 'knee extension',
            '/jointset/ankle_r/ankle_plantarflexion_r/value': 'ankle plantarflexion',
        }

        if not muscle_paths:
            muscle_paths = list()
            for muscle in model.getMuscleList():
                muscle_paths.append(muscle.getAbsolutePathString())
        num_muscles = len(muscle_paths)

        num_coordact = len(coordact_paths)

        time = moco_traj.getTimeMat()

        states_traj = moco_traj.exportToStatesTrajectory(model)

        # TODO for models without activation dynamics, we must prescribeControlsToModel().

        fig = pl.figure(figsize=(8.5, 11))
        tendon_forces = np.empty((len(time), num_muscles))
        for imusc, muscle_path in enumerate(muscle_paths):
            muscle = model.getComponent(muscle_path)
            for itime in range(len(time)):
                state = states_traj.get(itime)
                model.realizeDynamics(state)
                tendon_forces[itime, imusc] = muscle.getTendonForce(state)


        coordact_moments = np.empty((len(time), num_coordact))
        for ica, coordact_paths in enumerate(coordact_paths):
            coordact = model.getComponent(coordact_paths)
            for itime in range(len(time)):
                state = states_traj.get(itime)
                model.realizeDynamics(state)
                coordact_moments[itime, ica] = coordact.getActuation(state)


        for icoord, coord_path in enumerate(coord_paths):
            coord = model.getComponent(coord_path)

            label = os.path.split(coord_path)[-1] + '_moment'

            moment_arms = np.empty((len(time), num_muscles))
            for imusc, muscle_path in enumerate(muscle_paths):
                muscle = model.getComponent(muscle_path)
                for itime in range(len(time)):
                    state = states_traj.get(itime)
                    moment_arms[itime, imusc] = \
                        muscle.computeMomentArm(state, coord)

            ax = fig.add_subplot(num_coords, 1, icoord + 1)

            net_moment = np.zeros_like(time)
            for imusc, muscle_path in enumerate(muscle_paths):
                if np.any(moment_arms[:, imusc]) > 0.00001:
                    this_moment = tendon_forces[:, imusc] * moment_arms[:, imusc]
                    mom_integ = np.trapz(np.abs(this_moment), time)
                    net_moment += this_moment

            for ica, coordact_path in enumerate(coordact_paths):
                this_moment = coordact_moments[:, ica]
                ax.plot(time, this_moment, label=coordact_path)
                net_moment += this_moment

            if 'knee' in coord_path and knee_stiffness:
                knee_angle = moco_traj.getStateMat('/jointset/knee_r/knee_extension_r/value')
                spring_moment = -knee_stiffness * knee_angle
                ax.plot(time, spring_moment, label='spring')
                net_moment += spring_moment

            net_integ = np.trapz(np.abs(net_moment), x=time)
            sum_actuators_shown = np.zeros_like(time)
            for imusc, muscle_path in enumerate(muscle_paths):
                if np.any(moment_arms[:, imusc]) > 0.00001:
                    this_moment = tendon_forces[:, imusc] * moment_arms[:, imusc]
                    mom_integ = np.trapz(np.abs(this_moment), time)
                    if mom_integ > 0.01 * net_integ:
                        ax.plot(time, this_moment, label=muscle_path)

            ax.plot(time, net_moment,
                    label='net moment', color='black', linewidth=2)

            ax.set_title(coord_path)
            ax.set_ylabel('moment (N-m)')
            ax.legend(frameon=False, bbox_to_anchor=(1, 1),
                      loc='upper left', ncol=2)
            ax.tick_params(axis='both')
        ax.set_xlabel('time (% gait cycle)')

        fig.tight_layout()
        return fig

    def report_results(self, args):
        self.parse_args(args)

        fig = plt.figure(figsize=(7.5, 3))
        values = [
            '/jointset/hip_r/hip_extension_r/value',
            '/jointset/knee_r/knee_extension_r/value',
            '/jointset/ankle_r/ankle_plantarflexion_r/value',
        ]
        coord_signs = {
            '/jointset/hip_r/hip_extension_r/value': -1.0,
            '/jointset/knee_r/knee_extension_r/value': -1.0,
            '/jointset/ankle_r/ankle_plantarflexion_r/value': -1.0,
        }

        # TODO: Should be 3 muscles, all extensors.
        muscles = [
            ((0, 2), 'glut_max2', 'gluteus maximus'),
            ((1, 2), 'psoas', 'iliopsoas'),
            ((2, 2), 'semimem', 'hamstrings'),
            ((3, 2), 'vas_int', 'vasti'),
        ]
        grid = gridspec.GridSpec(4, 3)

        ax = fig.add_subplot(grid[0:4, 0:2])
        import cv2
        # Convert BGR color ordering to RGB.
        image = cv2.imread('squat_to_stand_visualization/'
                           'squat_to_stand_visualization.png')[...,::-1]
        ax.imshow(image)
        plt.axis('off')

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

        model = self.muscle_driven_model()
        assisted_model = self.assisted_model()
        assisted_model.updComponent('forceset/spring').setStiffness(stiffness)

        print(f'Stiffness: {stiffness}')
        with open('results/squat_to_stand_stiffness.txt', 'w') as f:
            f.write(f'{stiffness:.0f}')
        plot_solution(predict_solution, 'unassisted', color='gray')
        plot_solution(predict_assisted_solution, 'assisted')

        kinematics_rms = predict_solution.compareContinuousVariablesRMSPattern(
            predict_assisted_solution, 'states', '/jointset.*value$')
        kinematics_rms_deg = np.rad2deg(kinematics_rms)
        print('RMS difference in joint angles between conditions (degrees): '
              f'{kinematics_rms_deg}')
        with open('results/squat_to_stand_kinematics_rms.txt', 'w') as f:
            f.write(f'{kinematics_rms_deg:.1f}')

        fig.tight_layout() # w_pad=0.2)

        muscle_axes[0].legend(frameon=False, handlelength=1,
                             # bbox_to_anchor=(-1.0, 0.5),
                             # loc='center',
                             )

        fig.savefig('figures/squat_to_stand.png', dpi=600)

        coords = ['/jointset/hip_r/hip_extension_r',
                  '/jointset/knee_r/knee_extension_r',
                  '/jointset/ankle_r/ankle_plantarflexion_r']
        fig = self.plot_joint_moment_breakdown(model,
                                          predict_solution, coords)
        fig.savefig('figures/squat_to_stand_joint_moment_breakdown.png',
                    dpi=600)
        fig = self.plot_joint_moment_breakdown(assisted_model,
                                               predict_assisted_solution,
                                               coords,
                                               knee_stiffness=stiffness
                                               )
        fig.savefig('figures/squat_to_stand_assisted_'
                    'joint_moment_breakdown.png',
                    dpi=600)

        sol_predict_table = osim.TimeSeriesTable(self.predict_solution_file)
        sol_predict_duration = sol_predict_table.getTableMetaDataString('solver_duration')
        sol_predict_duration = float(sol_predict_duration) / 60.0
        print('prediction duration ', sol_predict_duration)
        with open('results/'
                  'squat_to_stand_predict_duration.txt', 'w') as f:
            f.write(f'{sol_predict_duration:.1f}')

        sol_predict_assisted_table = osim.TimeSeriesTable(self.predict_assisted_solution_file)
        sol_predict_assisted_duration = sol_predict_assisted_table.getTableMetaDataString('solver_duration')
        sol_predict_assisted_duration = float(sol_predict_assisted_duration) / 60.0
        print('prediction assisted duration ', sol_predict_assisted_duration)
        with open('results/'
                  'squat_to_stand_predict_assisted_duration.txt', 'w') as f:
            f.write(f'{sol_predict_assisted_duration:.1f}')

        # table = osim.analyze(model,
        #                      osim.MocoTrajectory(self.predict_solution_file),
        #              ['.*normalized_fiber_length'])
        # osim.STOFileAdapter.write(table,
        #                           'squat_to_stand_norm_fiber_length.sto')

        report = osim.report.Report(assisted_model,
                                    self.predict_assisted_solution_file,
                                    ref_files=[self.predict_solution_file])
        report.generate()


        self.create_ground_reaction_file(model, predict_solution,
                                         'results/squat_to_stand_ground_reaction.mot')

        self.create_ground_reaction_file(assisted_model, predict_assisted_solution,
                                         'results/squat_to_stand_assisted_ground_reaction.mot')

        # TODO: visualize in the GUI!
    def create_ground_reaction_file(self, model, solution, filepath):
        reaction = osim.analyzeSpatialVec(model, solution,
                                          ['/jointset/foot_ground_r\|reaction_on_child'])
        reaction = reaction.flatten(['_MX', '_MY', '_MZ', '_FX', '_FY', '_FZ'])
        prefix = '/jointset/foot_ground_r|reaction_on_child'
        FX = reaction.getDependentColumn(prefix + '_FX')
        FY = reaction.getDependentColumn(prefix + '_FY')
        FZ = reaction.getDependentColumn(prefix + '_FZ')
        MX = reaction.getDependentColumn(prefix + '_MX')
        MY = reaction.getDependentColumn(prefix + '_MY')
        MZ = reaction.getDependentColumn(prefix + '_MZ')

        # X coordinate of the center of pressure.
        x_cop = np.empty(reaction.getNumRows())
        z_cop = np.empty(reaction.getNumRows())
        TY = np.empty(reaction.getNumRows())
        state = model.initSystem()
        mass = model.getTotalMass(state)
        gravity_accel = model.get_gravity()
        mass_center = model.calcMassCenterPosition(state)
        print(f'com = {mass_center[0], mass_center[1], mass_center[2]}')
        print(f'weight = {mass * abs(gravity_accel[1])}')
        offset = model.getBodySet().get('calcn_r').getPositionInGround(state)
        x_offset = offset.get(0)
        z_offset = offset.get(2)
        print(f'x_offset = {x_offset}; z_offset = {z_offset}')
        for i in range(reaction.getNumRows()):
            x = MZ[i] / FY[i] + x_offset
            z = -MX[i] / FY[i] + z_offset
            x_cop[i] = x
            z_cop[i] = z
            TY[i] = MY[i] - (x - x_offset) * FZ[i] + (z - z_offset) * FX[i]

        zeroMatrix = osim.Matrix(reaction.getNumRows(), 1, 0)
        zero = osim.Vector(reaction.getNumRows(), 0)
        external_loads = osim.TimeSeriesTable(reaction.getIndependentColumn(),
                                              zeroMatrix, ['zero'])
        external_loads.appendColumn('ground_force_vx', FX)
        external_loads.appendColumn('ground_force_vy', FY)
        external_loads.appendColumn('ground_force_vz', FZ)
        external_loads.appendColumn('ground_force_px', osim.Vector(x_cop))
        external_loads.appendColumn('ground_force_py', zero)
        external_loads.appendColumn('ground_force_pz', osim.Vector(z_cop))
        external_loads.appendColumn('ground_torque_x', zero)
        external_loads.appendColumn('ground_torque_y', osim.Vector(TY))
        external_loads.appendColumn('ground_torque_z', zero)
        osim.STOFileAdapter.write(external_loads, filepath)
