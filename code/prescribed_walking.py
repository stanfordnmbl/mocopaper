import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities


class MotionPrescribedWalking(MocoPaperResult):
    def __init__(self):
        super(MotionPrescribedWalking, self).__init__()
        self.initial_time = 0.83 # 0.450
        self.final_time = 2.0 # 1.565
        # self.footstrike = 0.836 # 1.424
        self.mocoinverse_solution_file = \
            '%s/results/motion_prescribed_walking_inverse_solution.sto'
        self.mocoinverse_jointreaction_solution_file = \
            '%s/results/motion_prescribed_walking_inverse_jointreaction_solution.sto'
        self.side = 'l'

    def create_model_processor(self, root_dir):

        # Create base model without reserves.
        model = osim.Model(os.path.join(root_dir,
                'resources/Rajagopal2016/'
                'subject_walk_armless_contact_bounded_80musc.osim'))

        forceSet = model.getForceSet()
        numContacts = 0
        for i in np.arange(forceSet.getSize()):
            forceName = forceSet.get(int(i)).getName()
            if 'contact' in forceName: numContacts += 1

        print('Removing contact force elements from model...')
        contactsRemoved = 0
        while contactsRemoved < numContacts:
            for i in np.arange(forceSet.getSize()):
                forceName = forceSet.get(int(i)).getName()
                if 'contact' in forceName: 
                    print('  --> removed', forceSet.get(int(i)).getName())
                    forceSet.remove(int(i))
                    contactsRemoved += 1
                    break

        def add_reserve(model, coord, max_control):
            actu = osim.CoordinateActuator(coord)
            if coord.startswith('lumbar'):
                prefix = 'torque_'
            elif coord.startswith('pelvis'):
                prefix = 'residual_'
            else:
                prefix = 'reserve_'
            actu.setName(prefix + coord)
            actu.setMinControl(-1)
            actu.setMaxControl(1)
            actu.setOptimalForce(max_control)
            model.addForce(actu)
        add_reserve(model, 'lumbar_extension', 50)
        add_reserve(model, 'lumbar_bending', 50)
        add_reserve(model, 'lumbar_rotation', 20)
        add_reserve(model, 'pelvis_tilt', 60)
        add_reserve(model, 'pelvis_list', 30)
        add_reserve(model, 'pelvis_rotation', 15)
        add_reserve(model, 'pelvis_tx', 60)
        add_reserve(model, 'pelvis_ty', 200)
        add_reserve(model, 'pelvis_tz', 35)

        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        modelProcessor.append(osim.ModOpAddReserves(1, 15, True))
        baseModel = modelProcessor.process()


        # Create model for CMC
        modelProcessorCMC = osim.ModelProcessor(baseModel)
        cmcModel = modelProcessorCMC.process()

        tasks = osim.CMC_TaskSet()
        for coord in cmcModel.getCoordinateSet():
            cname = coord.getName()
            if cname.endswith('_beta'):
                continue
            task = osim.CMC_Joint()
            task.setName(cname)
            task.setCoordinateName(cname)
            task.setKP(400, 1, 1)
            task.setKV(80, 1, 1)
            task.setActive(True, False, False)
            tasks.cloneAndAppend(task)
        tasks.printToXML(os.path.join(root_dir, 'code',
                                      'motion_prescribed_walking_cmc_tasks.xml'))

        cmcModel.printToXML(os.path.join(root_dir, "resources/Rajagopal2016/"
                            "subject_walk_armless_for_cmc.osim"))

        # Create direct collocation model:
        #   - implicit tendon compliance mode
        #   - add external loads
        baseModel.initSystem()
        muscles = baseModel.updMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = muscles.get(int(imusc))
            if 'gas' in muscle.getName() or 'soleus' in muscle.getName():
                muscle.set_ignore_tendon_compliance(False)

        modelProcessorDC = osim.ModelProcessor(baseModel)
        modelProcessorDC.append(
            osim.ModOpTendonComplianceDynamicsModeDGF('implicit'))

        ext_loads_xml = os.path.join(root_dir,
                                     "resources/Rajagopal2016/grf_walk.xml")
        modelProcessorDC.append(osim.ModOpAddExternalLoads(ext_loads_xml))

        return modelProcessorDC

    def parse_args(self, args):
        self.cmc = False
        self.inverse = False
        self.knee = False
        if len(args) == 0:
            # self.cmc = True
            self.inverse = True
            self.knee = True
            return
        print('Received arguments {}'.format(args))
        if 'cmc' in args:
            self.cmc = True
        if 'inverse' in args:
            self.inverse = True
        if 'knee' in args:
            self.knee = True

    def generate_results(self, root_dir, args):
        self.parse_args(args)

        from transform_and_filter_ground_reaction import \
            transform_and_filter_ground_reaction
        # TODO: We aren't actually using this filtered file yet.
        transform_and_filter_ground_reaction(
            os.path.join(root_dir, 'resources/Rajagopal2016/emg_walk_raw.anc'),
            os.path.join(root_dir, 'resources/Rajagopal2016/grf_walk.mot'),
            'ground_reaction_forces',
            'rlr',
            mode='multicutoff')


        modelProcessor = self.create_model_processor(root_dir)

        coordinates = osim.TableProcessor(
            os.path.join(root_dir, "resources/Rajagopal2016/coordinates.mot"))
        coordinates.append(osim.TabOpLowPassFilter(6))
        coordinates.append(osim.TabOpUseAbsoluteStateNames())

        # cmc = osim.CMCTool()
        # cmc.setName('motion_prescribed_walking_cmc')
        # cmc.setExternalLoadsFileName('grf_walk.xml')
        # cmc.setDesiredKinematicsFileName('coordinates.mot')
        # # cmc.setLowpassCutoffFrequency(6)
        # cmc.printToXML('motion_prescribed_walking_cmc_setup.xml')
        cmc = osim.CMCTool(os.path.join(root_dir,
                                        'code/motion_prescribed_walking_cmc_setup.xml'))
        # 2.5 minute
        if self.cmc:
            cmc.run()

        # MocoInverse
        # -----------
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.05)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_convergence_tolerance(1e-2)
        inverse.set_reserves_weight(10.0)
  
        # 8 minutes
        if self.inverse:
            solution = inverse.solve()
            solution.getMocoSolution().write(self.mocoinverse_solution_file % root_dir)

        # MocoInverse, minimize joint reactions
        # -------------------------------------
        study = inverse.initialize()
        problem = study.updProblem()
        reaction_r = osim.MocoJointReactionGoal('reaction_r', 0.1)
        reaction_r.setJointPath('/jointset/walker_knee_r')
        reaction_r.setReactionMeasures(['force-x', 'force-y'])
        reaction_l = osim.MocoJointReactionGoal('reaction_l', 0.1)
        reaction_l.setJointPath('/jointset/walker_knee_l')
        reaction_l.setReactionMeasures(['force-x', 'force-y'])
        problem.addGoal(reaction_r)
        problem.addGoal(reaction_l)
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

        # 13 minutes.
        if self.knee:
            solution_reaction = study.solve()
            solution_reaction.write(self.mocoinverse_jointreaction_solution_file % root_dir)

    def calc_reserves(self, root_dir, solution):
        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        output = osim.analyze(model, solution, ['.*reserve.*actuation'])
        return output

    def calc_max_knee_reaction_force(self, root_dir, solution):
        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        jr = osim.analyzeSpatialVec(model, solution,
                                    ['.*walker_knee.*reaction_on_parent.*'])
        jr = jr.flatten(['_mx', '_my', '_mz', '_fx', '_fy', '_fz'])
        max = -np.inf
        for itime in range(jr.getNumRows()):
            for irxn in range(int(jr.getNumColumns() / 6)):
                fx = jr.getDependentColumnAtIndex(6 * irxn + 3)[itime]
                fy = jr.getDependentColumnAtIndex(6 * irxn + 4)[itime]
                fz = jr.getDependentColumnAtIndex(6 * irxn + 5)[itime]
                norm = np.sqrt(fx**2 + fy**2 + fz**2)
                max = np.max([norm, max])
        g = np.abs(model.get_gravity()[1])
        state = model.initSystem()
        mass = model.getTotalMass(state)
        weight = mass * g
        return max / weight

    def report_results(self, root_dir, args):
        self.parse_args(args)

        if self.inverse:
            sol_inverse_table = osim.TimeSeriesTable(self.mocoinverse_solution_file % root_dir)
            inverse_duration = sol_inverse_table.getTableMetaDataString('solver_duration')
            inverse_duration = float(inverse_duration) / 60.0
            print('inverse duration ', inverse_duration)
            with open(os.path.join(root_dir, 'results/'
                      'motion_prescribed_walking_inverse_duration.txt'), 'w') as f:
                f.write(f'{inverse_duration:.1f}')

        if self.knee:
            sol_inverse_jr_table = osim.TimeSeriesTable(self.mocoinverse_jointreaction_solution_file % root_dir)
            inverse_jr_duration = sol_inverse_jr_table.getTableMetaDataString('solver_duration')
            inverse_jr_duration = float(inverse_jr_duration) / 60.0
            print('inverse joint reaction duration ', inverse_jr_duration)
            with open(os.path.join(root_dir, 'results/'
                      'motion_prescribed_walking_inverse_jr_duration.txt'), 'w') as f:
                f.write(f'{inverse_jr_duration:.1f}')

        emg = self.load_electromyography(root_dir)

        if self.inverse:
            sol_inverse = osim.MocoTrajectory(self.mocoinverse_solution_file % root_dir)
            time_inverse = sol_inverse.getTimeMat()

        if self.knee and self.inverse:
            sol_inverse_jointreaction = \
                osim.MocoTrajectory(self.mocoinverse_jointreaction_solution_file % root_dir)
            sol_inverse_jointreaction.insertStatesTrajectory(
                sol_inverse.exportToStatesTable(), False)
            mocoinverse_jr_solution_file = \
                self.mocoinverse_jointreaction_solution_file.replace('.sto',
                                                                     '_with_q_u.sto')

            sol_inverse_jointreaction.write(mocoinverse_jr_solution_file % root_dir)
            time_inverse_jointreaction = sol_inverse_jointreaction.getTimeMat()

        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        model.initSystem()
        print(f'Degrees of freedom: {model.getCoordinateSet().getSize()}')

        # TODO: slight shift in CMC solution might be due to how we treat
        # percent gait cycle and the fact that CMC is missing 0.02 seconds.
        if self.cmc:
            cmc_states_fpath = os.path.join(root_dir,
                                            'results/motion_prescribed_walking_cmc_results/'
                                            'motion_prescribed_walking_cmc_states.sto')
            sol_cmc = osim.TimeSeriesTable(cmc_states_fpath)
            time_cmc = np.array(sol_cmc.getIndependentColumn())

        plot_breakdown = True

        if plot_breakdown:
            fig = utilities.plot_joint_moment_breakdown(model,
                                                        sol_inverse,
                                                        ['/jointset/hip_l/hip_flexion_l',
                                                         '/jointset/hip_l/hip_adduction_l',
                                                         '/jointset/hip_l/hip_rotation_l',
                                                         '/jointset/walker_knee_l/knee_angle_l',
                                                         '/jointset/ankle_l/ankle_angle_l'],
                                                        )
            fig.savefig(os.path.join(root_dir, 'results/motion_prescribed_walking_inverse_'
                        'joint_moment_breakdown.png'),
                        dpi=600)

        # report = osim.report.Report(model, mocoinverse_jr_solution_file % root_dir)
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
            fig.savefig(os.path.join(root_dir,
                                     'results/motion_prescribed_walking_inverse_'
                                     'jointreaction_joint_moment_breakdown.png'),
                        dpi=600)

        coords = [
            (f'/jointset/ground_pelvis/pelvis_tx', 'pelvis tx', 1.0),
            (f'/jointset/ground_pelvis/pelvis_ty', 'pelvis ty', 1.0),
            (f'/jointset/ground_pelvis/pelvis_tz', 'pelvis tz', 1.0),
            (f'/jointset/ground_pelvis/pelvis_rotation', 'pelvis rotation', 1.0),
            (f'/jointset/ground_pelvis/pelvis_list', 'pelvis list', 1.0),
            (f'/jointset/ground_pelvis/pelvis_tilt', 'pelvis tilt', 1.0),
            (f'/jointset/back/lumbar_rotation', 'lumbar rotation', 1.0),
            (f'/jointset/back/lumbar_extension', 'lumbar extension', 1.0),
            (f'/jointset/back/lumbar_bending', 'lumbar bending', 1.0),
            (f'/jointset/hip_{self.side}/hip_adduction_{self.side}',
                'hip adduction', 1.0),
            (f'/jointset/hip_{self.side}/hip_rotation_{self.side}',
             'hip rotation', 1.0),
            (
            f'/jointset/hip_{self.side}/hip_flexion_{self.side}', 'hip flexion',
            1.0),
            (f'/jointset/walker_knee_{self.side}/knee_angle_{self.side}',
             'knee flexion', 1.0),
            (f'/jointset/ankle_{self.side}/ankle_angle_{self.side}',
             'ankle plantarflexion', 1.0),
        ]
        from utilities import toarray
        fig = plt.figure(figsize=(4, 8))

        for ic, coord in enumerate(coords):
            ax = plt.subplot(7, 2, ic + 1)
            if self.cmc:
                y_cmc = coord[2] * np.rad2deg(
                    toarray(sol_cmc.getDependentColumn(f'{coord[0]}/value')),)
                self.plot(ax, time_cmc, y_cmc, label='CMC', color='k',
                          linewidth=2)

            if self.inverse:
                y_inverse = coord[2] * np.rad2deg(
                    sol_inverse.getStateMat(f'{coord[0]}/value'))
                self.plot(ax, time_inverse, y_inverse, label='MocoInverse',
                          linewidth=2)

            ax.plot([0], [0], label='MocoInverse-knee', linewidth=2)

            # if self.track:
            #     y_track = coord[2] * np.rad2deg(
            #         sol_track.getStateMat(f'{coord[0]}/value'))
            #     self.plot(ax, time_track, y_track, label='MocoTrack',
            #               linewidth=1)

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

        fig.savefig(os.path.join(root_dir,
                                 'figures/motion_prescribed_walking_kinematics.png'),
                    dpi=600)


        fig = plt.figure(figsize=(7.5, 3.3))
        gs = gridspec.GridSpec(3, 4, width_ratios=[0.8, 1, 1, 1])

        ax = fig.add_subplot(gs[0:3, 0])
        import cv2
        # Convert BGR color ordering to RGB.
        image = cv2.imread(os.path.join(root_dir,
                                        'figures/motion_prescribed_walking_inverse_model.png'))[
                ..., ::-1]
        ax.imshow(image)
        plt.axis('off')


        muscles = [
            ((0, 0), 'glmax2', 'gluteus maximus', 'GMAX'),
            ((0, 1), 'psoas', 'psoas', ''),
            # TODO: dashed semimem
            ((1, 0), 'semiten', 'semitendinosus', 'MH'),
            ((0, 2), 'recfem', 'rectus femoris', 'RF'),
            ((1, 1), 'bfsh', 'biceps femoris short head', 'BF'),
            ((1, 2), 'vaslat', 'vastus lateralis', 'VL'),
            ((2, 0), 'gasmed', 'medial gastrocnemius', 'GAS'),
            ((2, 1), 'soleus', 'soleus', 'SOL'),
            ((2, 2), 'tibant', 'tibialis anterior', 'TA'),
        ]
        legend_handles_and_labels = []
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[muscle[0][0], muscle[0][1] + 1])
            activation_path = f'/forceset/{muscle[1]}_{self.side}/activation'
            legend_musc = []
            # if self.cmc:
            #     cmc_activ = toarray(sol_cmc.getDependentColumn(activation_path))
            #     handle, = self.plot(ax, time_cmc,
            #               cmc_activ,
            #               linewidth=1,
            #               color='black',
            #               label='Computed Muscle Control',
            #               )
            #     legend_musc.append((handle, 'Computed Muscle Control'))
            if self.inverse:
                inverse_activ = sol_inverse.getStateMat(activation_path)
                handle, = self.plot(ax, time_inverse, inverse_activ,
                          label='MocoInverse',
                          color='black',
                          linewidth=2, zorder=2)
                legend_musc.append((handle, 'MocoInverse'))
            if self.knee and self.inverse:
                handle, = self.plot(ax, time_inverse_jointreaction,
                          sol_inverse_jointreaction.getStateMat(activation_path),
                          label='MocoInverse, knee',
                          linewidth=1, zorder=3)
                legend_musc.append((handle, 'MocoInverse, knee'))
            if self.inverse and len(muscle[3]) > 0:
                handle = self.plot(ax, emg['time'],
                                   emg[muscle[3]] * np.max(inverse_activ),
                                   shift=False,
                                   fill=True,
                                   color='lightgray',
                                   label='electromyography')
                legend_musc.append((handle,
                                    'electromyography (normalized; peak '
                                    'matches the peak from MocoInverse)'))
            if im == 0:
                legend_handles_and_labels = legend_musc
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

        # if self.inverse and self.knee:
        legend_handles, legend_labels = zip(*legend_handles_and_labels)
        plt.figlegend(legend_handles, legend_labels,
            frameon=False,
            ncol=5,
            loc='lower center',
        )
        fig.tight_layout(h_pad=1, rect=(0, 0.07, 1, 1), pad=0.4)

        fig.savefig(
            os.path.join(root_dir, 'figures/motion_prescribed_walking.png'),
            dpi=600)

        if self.cmc and self.inverse:
            mocosol_cmc = sol_inverse.clone()
            mocosol_cmc.insertStatesTrajectory(sol_cmc, True)
            inv_sol_rms = \
                sol_inverse.compareContinuousVariablesRMSPattern(mocosol_cmc,
                                                                 'states',
                                                                 '.*activation')

            peak_inverse_activation = -np.inf
            for state_name in sol_inverse.getStateNames():
                if state_name.endswith('activation'):
                    column = sol_inverse.getStateMat(state_name)
                    peak_inverse_activation = np.max([peak_inverse_activation,
                                                      np.max(column)])

            inv_sol_rms_pcent = 100.0 * inv_sol_rms / peak_inverse_activation
            print(f'Peak MocoInverse activation: {peak_inverse_activation}')
            print('CMC MocoInverse activation RMS: ', inv_sol_rms_pcent)
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_cmc_rms.txt'), 'w') as f:
                f.write(f'{inv_sol_rms_pcent:.0f}')

        if self.inverse:
            res_inverse = self.calc_reserves(root_dir, sol_inverse)
            column_labels = res_inverse.getColumnLabels()
            max_res_inverse = -np.inf
            for icol in range(res_inverse.getNumColumns()):
                column = utilities.toarray(
                    res_inverse.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                max_res_inverse = np.max([max_res_inverse, max])
                print(f'inverse max abs {column_labels[icol]}: {max}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_max_reserve.txt'), 'w') as f:
                f.write(f'{max_res_inverse:.1f}')

        if self.knee and self.inverse:
            res_inverse_jr = self.calc_reserves(root_dir,
                                                sol_inverse_jointreaction)
            column_labels = res_inverse_jr.getColumnLabels()
            max_res_inverse_jr = -np.inf
            for icol in range(res_inverse_jr.getNumColumns()):
                column = utilities.toarray(
                    res_inverse_jr.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                max_res_inverse_jr = np.max([max_res_inverse_jr, max])
                print(f'inverse_jr max abs {column_labels[icol]}: {max}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_jr_max_reserve.txt'), 'w') as f:
                f.write(f'{max_res_inverse_jr:.3f}')

            maxjr_inverse = self.calc_max_knee_reaction_force(root_dir, sol_inverse)
            maxjr_inverse_jr = self.calc_max_knee_reaction_force(root_dir, sol_inverse_jointreaction)
            print(f'Max joint reaction {maxjr_inverse} -> {maxjr_inverse_jr}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_maxjr.txt'), 'w') as f:
                f.write(f'{maxjr_inverse:.1f}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_jr_maxjr.txt'), 'w') as f:
                f.write(f'{maxjr_inverse_jr:.1f}')

        if self.inverse:
            states = sol_inverse.exportToStatesTrajectory(model)
            duration = sol_inverse.getFinalTime() - sol_inverse.getInitialTime()
            avg_speed = (model.calcMassCenterPosition(states[states.getSize() - 1])[0] -
                         model.calcMassCenterPosition(states[0])[0]) / duration
            print(f'Average speed: {avg_speed}')

        if self.inverse:
            report = osim.report.Report(model, self.mocoinverse_solution_file % root_dir)
            report.generate()

        if self.knee and self.inverse:
            report = osim.report.Report(model,
                                        mocoinverse_jr_solution_file % root_dir)
            report.generate()

