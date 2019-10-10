import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities

class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        self.initial_time = 0.83 # 0.450
        self.final_time = 2.0 # 1.565
        # self.footstrike = 0.836 # 1.424
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
            starting_time = self.initial_time
            # starting_time = self.footstrike
        return utilities.shift_data_to_cycle(initial_time, final_time,
                                             starting_time, time, y, cut_off=True)

    def create_model_processor(self):

        # Create CMC model first.
        # TODO: try 18 muscles first for CMC.
        modelProcessorCMC = osim.ModelProcessor(
            "resources/Rajagopal2016/subject_walk_armless_18musc.osim")
            # "resources/Rajagopal2016/subject_walk_armless_80musc.osim")
        modelProcessorCMC.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessorCMC.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessorCMC.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        modelProcessorCMC.append(osim.ModOpIgnoreTendonCompliance())
        # modelProcessorCMC.append(osim.ModOpAddReserves(50, 1))
        modelProcessorCMC.append(
            osim.ModOpTendonComplianceDynamicsModeDGF('explicit'))
        # modelProcessorCMC.append(osim.ModOpScaleTendonSlackLength(0.96))

        cmcModel = modelProcessorCMC.process()
        cmcModel.initSystem()
        muscles = cmcModel.updMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = osim.DeGrooteFregly2016Muscle.safeDownCast(
                muscles.get(int(imusc)))
            muscle.set_fiber_damping(0)
            #muscle.set_clamp_normalized_tendon_length(True)

        tasks = osim.CMC_TaskSet()
        for coord in cmcModel.getCoordinateSet():
            if not coord.getName().endswith('_beta'):
                task = osim.CMC_Joint()
                task.setName(coord.getName())
                task.setCoordinateName(coord.getName())
                task.setKP(100, 1, 1)
                task.setKV(20, 1, 1)
                task.setActive(True, False, False)
                tasks.cloneAndAppend(task)
        tasks.printToXML('motion_tracking_walking_cmc_tasks.xml')

        cmcModel.printToXML("resources/Rajagopal2016/"
                            "subject_walk_armless_for_cmc.osim")

        # Add external loads to MocoTrack model.
        ext_loads_xml = "resources/Rajagopal2016/grf_walk.xml"
        modelProcessorDC = osim.ModelProcessor(cmcModel)
        modelProcessorDC.append(osim.ModOpAddExternalLoads(ext_loads_xml))
        modelProcessorDC.append(
            osim.ModOpTendonComplianceDynamicsModeDGF('implicit'))
        return modelProcessorDC

    def parse_args(self, args):
        self.cmc = False
        self.track = False
        self.inverse = False
        self.knee = False
        if len(args) == 0:
            self.cmc = True
            self.track = True
            self.inverse = True
            self.knee = True
            return
        print('Received arguments {}'.format(args))
        if 'cmc' in args:
            self.cmc = True
        if 'track' in args:
            self.track = True
        if 'inverse' in args:
            self.inverse = True
        if 'knee' in args:
            self.knee = True

    def generate_results(self, args):
        self.parse_args(args)

        from transform_and_filter_ground_reaction import \
            transform_and_filter_ground_reaction
        # TODO: We aren't actually using this filtered file yet.
        transform_and_filter_ground_reaction(
            'resources/Rajagopal2016/emg_walk_raw.anc',
            'resources/Rajagopal2016/grf_walk.mot',
            'ground_reaction_forces',
            'rlr',
            mode='multicutoff')


        modelProcessor = self.create_model_processor()

        # TODO:
        #  - avoid removing muscle passive forces
        #  - why does soleus activation drop so quickly?
        #  - set MocoTrack initial guess to the MocoInverse solution?


        # TODO: why is recfem used instead of vaslat? recfem counters the hip
        # extension moment in early stance.

        coordinates = osim.TableProcessor(
            "resources/Rajagopal2016/coordinates.mot")
        coordinates.append(osim.TabOpLowPassFilter(6))
        coordinates.append(osim.TabOpUseAbsoluteStateNames())

        # # TODO plotting should happen separately from generating the results.
        # cmc = osim.CMCTool()
        # cmc.setName('motion_tracking_walking_cmc')
        # cmc.setExternalLoadsFileName('grf_walk.xml')
        # # TODO filter:
        # cmc.setDesiredKinematicsFileName('coordinates.mot')
        # # cmc.setLowpassCutoffFrequency(6)
        # cmc.printToXML('motion_tracking_walking_cmc_setup.xml')
        cmc = osim.CMCTool('motion_tracking_walking_cmc_setup.xml')
        # 2.5 minute
        if self.cmc:
            cmc.run()

        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.05)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_tolerance(1e-3)
        # inverse.set_reserves_weight(10.0)
        # 8 minutes
        if self.inverse:
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


        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        # TODO: try 1e-2 for MocoInverse without JR minimization.
        solver.set_optim_convergence_tolerance(1e-2)

        # 50 minutes.
        if self.knee:
            solution_reaction = study.solve()
            solution_reaction.write(self.mocoinverse_jointreaction_solution_file)


        # Create and name an instance of the MocoTrack tool.
        track = osim.MocoTrack()
        track.setName("motion_tracking_walking")
        track.setModel(modelProcessor)
        track.setStatesReference(coordinates)
        track.set_states_global_tracking_weight(0.05)

        # This setting allows extra data columns contained in the states
        # reference that don't correspond to model coordinates.
        track.set_allow_unused_references(True)

        track.set_track_reference_position_derivatives(True)

        # TODO: Use MocoInverse solution as initial guess for MocoTrack.
        track.set_apply_tracked_states_to_guess(True)

        # Initial time, final time, and mesh interval.
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(0.01)

        moco = track.initialize()
        moco.set_write_solution("results/")

        problem = moco.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal("control_effort"))
        effort.setWeightForControlPattern('.*reserve_.*', 50)
        effort.setWeightForControlPattern('.*reserve_pelvis_.*', 10)

        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.set_optim_convergence_tolerance(1e-4)

        # Solve and visualize.
        moco.printToXML('motion_tracking_walking.omoco')
        # 45 minutes
        if self.track:
            solution = moco.solve()
            solution.write(self.mocotrack_solution_file)
            # moco.visualize(solution)

    def plot(self, ax, time, y, shift=True, fill=False, *args, **kwargs):
        if shift:
            shifted_time, shifted_y = self.shift(time, y)
        else:
            duration = self.final_time - self.initial_time
            shifted_time, shifted_y = self.shift(time, y,
                                                 starting_time=self.initial_time + 0.5 * duration)

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

    def report_results(self, args):
        self.parse_args(args)

        if self.track:
            sol_track_table = osim.TimeSeriesTable(self.mocotrack_solution_file)
            track_duration = sol_track_table.getTableMetaDataString('solver_duration')
            track_duration = float(track_duration) / 60.0 / 60.0
            print('track duration ', track_duration)
            with open('results/'
                      'motion_tracking_walking_track_duration.txt', 'w') as f:
                f.write(f'{track_duration:.1f}')

        if self.inverse:
            sol_inverse_table = osim.TimeSeriesTable(self.mocoinverse_solution_file)
            inverse_duration = sol_inverse_table.getTableMetaDataString('solver_duration')
            inverse_duration = float(inverse_duration) / 60.0
            print('inverse duration ', inverse_duration)
            with open('results/'
                      'motion_tracking_walking_inverse_duration.txt', 'w') as f:
                f.write(f'{inverse_duration:.1f}')

        if self.knee:
            sol_inverse_jr_table = osim.TimeSeriesTable(self.mocoinverse_jointreaction_solution_file)
            inverse_jr_duration = sol_inverse_jr_table.getTableMetaDataString('solver_duration')
            inverse_jr_duration = float(inverse_jr_duration) / 60.0
            print('inverse joint reaction duration ', inverse_jr_duration)
            with open('results/'
                      'motion_tracking_walking_inverse_jr_duration.txt', 'w') as f:
                f.write(f'{inverse_jr_duration:.1f}')

        emg = self.load_electromyography()


        if self.track:
            sol_track = osim.MocoTrajectory(self.mocotrack_solution_file)
            time_track = sol_track.getTimeMat()

        if self.inverse:
            sol_inverse = osim.MocoTrajectory(self.mocoinverse_solution_file)
            time_inverse = sol_inverse.getTimeMat()

        if self.knee:
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
        if self.cmc:
            sol_cmc = osim.TimeSeriesTable('results/motion_tracking_walking_cmc_results/'
                                           'motion_tracking_walking_cmc_states.sto')
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
        # muscles = [
        #     ((0, 0), 'glut_max2', 'gluteus maximus', 'GMAX'),
        #     ((0, 1), 'psoas', 'psoas', ''),
        #     ((1, 0), 'semimem', 'semimembranosus', 'MH'),
        #     ((0, 2), 'rect_fem', 'rectus femoris', 'RF'),
        #     ((1, 1), 'bifemsh', 'biceps femoris short head', 'BF'),
        #     ((1, 2), 'vas_int', 'vastus lateralis', 'VL'),
        #     ((2, 0), 'med_gas', 'medial gastrocnemius', 'GAS'),
        #     ((2, 1), 'soleus', 'soleus', 'SOL'),
        #     ((2, 2), 'tib_ant', 'tibialis anterior', 'TA'),
        # ]
        muscles = [
            ((0, 0), 'glmax2', 'gluteus maximus', 'GMAX'),
            ((0, 1), 'psoas', 'psoas', ''),
            ((1, 0), 'semimem', 'semimembranosus', 'MH'),
            ((0, 2), 'recfem', 'rectus femoris', 'RF'),
            ((1, 1), 'bfsh', 'biceps femoris short head', 'BF'),
            ((1, 2), 'vasint', 'vastus lateralis', 'VL'),
            ((2, 0), 'gasmed', 'medial gastrocnemius', 'GAS'),
            ((2, 1), 'soleus', 'soleus', 'SOL'),
            ((2, 2), 'tibant', 'tibialis anterior', 'TA'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[muscle[0][0], muscle[0][1]])
            activation_path = f'/forceset/{muscle[1]}_{self.side}/activation'
            if self.cmc:
                cmc_activ = toarray(sol_cmc.getDependentColumn(activation_path))
                self.plot(ax, time_cmc,
                          cmc_activ,
                          linewidth=3,
                          label='CMC',
                          )
            if self.track:
                self.plot(ax, time_track, sol_track.getStateMat(activation_path),
                          label='MocoTrack',
                          linewidth=2)
            if self.inverse:
                self.plot(ax, time_inverse,
                          sol_inverse.getStateMat(activation_path),
                          label='MocoInverse',
                          linewidth=2)
            if self.knee:
                self.plot(ax, time_inverse_jointreaction,
                          sol_inverse_jointreaction.getStateMat(activation_path),
                          label='MocoInverse, knee',
                          linewidth=2)
            if self.cmc and len(muscle[3]) > 0:
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

        if self.cmc and self.inverse:
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

        if self.track:
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

        if self.inverse:
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

        if self.knee:
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

            maxjr_inverse = self.calc_max_knee_reaction_force(sol_inverse)
            maxjr_inverse_jr = self.calc_max_knee_reaction_force(sol_inverse_jointreaction)
            print(f'Max joint reaction {maxjr_inverse} -> {maxjr_inverse_jr}')
            with open('results/motion_tracking_walking_'
                      'inverse_maxjr.txt', 'w') as f:
                f.write(f'{maxjr_inverse:.1f}')
            with open('results/motion_tracking_walking_'
                      'inverse_jr_maxjr.txt', 'w') as f:
                f.write(f'{maxjr_inverse_jr:.1f}')

        if self.inverse:
            states = sol_inverse.exportToStatesTrajectory(model)
            duration = sol_inverse.getFinalTime() - sol_inverse.getInitialTime()
            avg_speed = (model.calcMassCenterPosition(states[states.getSize() - 1])[0] -
                         model.calcMassCenterPosition(states[0])[0]) / duration
            print(f'Average speed: {avg_speed}')


