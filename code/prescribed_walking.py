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
                'subject_walk_contact_bounded_80musc.osim'))

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

        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l', 'radius_hand_r',
             'radius_hand_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        modelProcessor.append(osim.ModOpFiberDampingDGF(0))
        baseModel = modelProcessor.process()

        # Create model for CMC
        modelProcessorCMC = osim.ModelProcessor(baseModel)
        modelProcessorCMC.append(osim.ModOpAddReserves(1))
        cmcModel = modelProcessorCMC.process()

        tasks = osim.CMC_TaskSet()
        for coord in cmcModel.getCoordinateSet():
            cname = coord.getName()
            if cname.endswith('_beta'):
                continue
            task = osim.CMC_Joint()
            task.setName(cname)
            task.setCoordinateName(cname)
            task.setKP(100, 1, 1)
            task.setKV(20, 1, 1)
            task.setActive(True, False, False)
            tasks.cloneAndAppend(task)
        tasks.printToXML(
            os.path.join(root_dir, 'code',
                         'motion_prescribed_walking_cmc_tasks.xml'))

        cmcModel.printToXML(os.path.join(root_dir, 'resources/Rajagopal2016/'
                                         'subject_walk_for_cmc.osim'))

        # Create direct collocation model:
        #   - TODO reserves
        #   - implicit tendon compliance mode
        #   - add external loads
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
        add_reserve(baseModel, 'lumbar_extension', 50)
        add_reserve(baseModel, 'lumbar_bending', 50)
        add_reserve(baseModel, 'lumbar_rotation', 20)
        for side in ['_l', '_r']:
            add_reserve(baseModel, f'arm_flex{side}', 15)
            add_reserve(baseModel, f'arm_add{side}', 15)
            add_reserve(baseModel, f'arm_rot{side}', 5)
            add_reserve(baseModel, f'elbow_flex{side}', 5)
            add_reserve(baseModel, f'pro_sup{side}', 1)
            add_reserve(baseModel, f'hip_rotation{side}', 0.5)

        add_reserve(baseModel, 'pelvis_tilt', 60)
        add_reserve(baseModel, 'pelvis_list', 30)
        add_reserve(baseModel, 'pelvis_rotation', 15)
        add_reserve(baseModel, 'pelvis_tx', 100)
        add_reserve(baseModel, 'pelvis_ty', 200)
        add_reserve(baseModel, 'pelvis_tz', 35)
        baseModel.initSystem()
        muscles = baseModel.updMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = muscles.get(int(imusc))
            if 'gas' in muscle.getName() or 'soleus' in muscle.getName():
                muscle.set_ignore_tendon_compliance(False)

        modelProcessorDC = osim.ModelProcessor(baseModel)
        modelProcessorDC.append(osim.ModOpFiberDampingDGF(0.01))
        modelProcessorDC.append(osim.ModOpAddReserves(1, 2.5, True))
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
            self.cmc = True
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

    def create_inverse(self, root_dir, modelProcessor, mesh_interval):
        coordinates = osim.TableProcessor(
            os.path.join(root_dir, "resources/Rajagopal2016/coordinates.mot"))
        coordinates.append(osim.TabOpLowPassFilter(6))
        coordinates.append(osim.TabOpUseAbsoluteStateNames())
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(mesh_interval)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_convergence_tolerance(1e-2) # TODO
        inverse.set_mesh_interval(mesh_interval)
        return inverse

    def solve_inverse(self, root_dir, modelProcessor, mesh_interval,
                      solution_filepath):
        inverse = self.create_inverse(root_dir, modelProcessor, mesh_interval)
        solution = inverse.solve()
        solution.getMocoSolution().write(solution_filepath)

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

        cmc = osim.CMCTool(
            os.path.join(root_dir,
                         'code/motion_prescribed_walking_cmc_setup.xml'))
        analyze = osim.AnalyzeTool(
            os.path.join(root_dir,
                         'code/motion_prescribed_walking_analyze_setup.xml'))
        # 2.5 minute
        if self.cmc:
            cmc.run()
            analyze.run()

        mesh_interval = (self.final_time - self.initial_time) / 25.0

        # 8 minutes
        if self.inverse:
            solution_filepath = self.mocoinverse_solution_file % root_dir
            self.solve_inverse(root_dir, modelProcessor, mesh_interval,
                               solution_filepath)
            # TODO
            # # 0.04 and 0.03 work
            # inverse.set_mesh_interval(0.01)
            # # inverse.set_convergence_tolerance(1e-4)
            # solution = inverse.solve()
            # solution.getMocoSolution().write(self.mocoinverse_fine_solution_file % root_dir)

        # MocoInverse, minimize joint reactions
        # -------------------------------------
        inverse = self.create_inverse(root_dir, modelProcessor, mesh_interval)
        study = inverse.initialize()
        problem = study.updProblem()
        reaction_r = osim.MocoJointReactionGoal('reaction_r', 0.005)
        reaction_r.setJointPath('/jointset/walker_knee_r')
        reaction_r.setReactionMeasures(['force-x', 'force-y'])
        reaction_l = osim.MocoJointReactionGoal('reaction_l', 0.005)
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

    def calc_knee_reaction_force(self, root_dir, solution):
        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        jr = osim.analyzeSpatialVec(model, solution,
                                    ['.*walker_knee.*reaction_on_parent.*'])
        jr = jr.flatten(['_mx', '_my', '_mz', '_fx', '_fy', '_fz'])
        traj = np.empty(jr.getNumRows())
        max = -np.inf
        for itime in range(jr.getNumRows()):
            for irxn in range(int(jr.getNumColumns() / 6)):
                fx = jr.getDependentColumnAtIndex(6 * irxn + 3)[itime]
                fy = jr.getDependentColumnAtIndex(6 * irxn + 4)[itime]
                fz = jr.getDependentColumnAtIndex(6 * irxn + 5)[itime]
                norm = np.sqrt(fx**2 + fy**2 + fz**2)
                traj[itime] = norm
                max = np.max([norm, max])
        time = jr.getIndependentColumn()
        avg = np.trapz(traj, x=time) / (time[-1] - time[0])
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(time, traj)
        # plt.show()
        g = np.abs(model.get_gravity()[1])
        state = model.initSystem()
        mass = model.getTotalMass(state)
        weight = mass * g
        return max / weight, avg / weight

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
        emgPerry = self.load_electromyography_PerryBurnfield(root_dir)

        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        model.initSystem()
        numConstraints = 2
        numDOFs = model.getCoordinateSet().getSize() - numConstraints
        print(f'Degrees of freedom: {numDOFs}')

        if self.inverse:
            sol_inverse = osim.MocoTrajectory(
                self.mocoinverse_solution_file % root_dir)
            time_inverse = sol_inverse.getTimeMat()
            most_neg = self.calc_negative_muscle_forces_base(model, sol_inverse)
            if most_neg < -0.005:
                raise Exception("Muscle forces are too negative! sol_inverse")

        if self.knee and self.inverse:
            sol_inverse_jointreaction = \
                osim.MocoTrajectory(self.mocoinverse_jointreaction_solution_file % root_dir)
            sol_inverse_jointreaction.insertStatesTrajectory(
                sol_inverse.exportToStatesTable(), False)

            most_neg = self.calc_negative_muscle_forces_base(
                model, sol_inverse_jointreaction)
            if most_neg < -0.005:
                raise Exception(
                    "Muscle forces are too negative! sol_inverse_jointreaction")

            mocoinverse_jr_solution_file = \
                self.mocoinverse_jointreaction_solution_file.replace('.sto',
                                                                     '_with_q_u.sto')

            sol_inverse_jointreaction.write(mocoinverse_jr_solution_file % root_dir)
            time_inverse_jointreaction = sol_inverse_jointreaction.getTimeMat()

        plot_breakdown = False

        coords = ['/jointset/hip_l/hip_flexion_l',
                  '/jointset/hip_l/hip_adduction_l',
                  '/jointset/hip_l/hip_rotation_l',
                  '/jointset/walker_knee_l/knee_angle_l',
                  '/jointset/ankle_l/ankle_angle_l']
        if plot_breakdown and self.inverse:
            fig = utilities.plot_joint_moment_breakdown(model,
                                                        sol_inverse,
                                                        coords
                                                        )
            fig.savefig(os.path.join(root_dir, 'results/motion_prescribed_walking_inverse_'
                        'joint_moment_breakdown.png'),
                        dpi=600)

        # report = osim.report.Report(model, mocoinverse_jr_solution_file % root_dir)
        # report.generate()

        if plot_breakdown and self.knee:
            fig = utilities.plot_joint_moment_breakdown(model,
                                                        sol_inverse_jointreaction,
                                                        coords
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

            if self.inverse:
                y_inverse = coord[2] * np.rad2deg(
                    sol_inverse.getStateMat(f'{coord[0]}/value'))
                self.plot(ax, time_inverse, y_inverse, label='MocoInverse',
                          linewidth=2)

            ax.plot([0], [0], label='MocoInverse-knee', linewidth=2)

            ax.set_xlim(0, 100)
            if ic == 1:
                ax.legend(frameon=False, handlelength=1.9)
            if ic < len(coords) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
                ax.get_xaxis().set_label_coords(0.5, 0)
                # The '0' would intersect with the y-axis, so remove it.
                ax.set_xticks([0, 50, 100])
                ax.set_xticklabels(['', '50', '100'])
            ax.set_ylabel(f'{coord[1]} (degrees)')
            ax.get_yaxis().set_label_coords(-0.15, 0.5)

            ax.spines['bottom'].set_position('zero')
            utilities.publication_spines(ax)

        fig.savefig(os.path.join(root_dir,
                                 'figures/motion_prescribed_walking_kinematics.png'),
                    dpi=600)

        # Fig7
        # ----
        fig = plt.figure(figsize=(7.5, 3.3))
        gs = gridspec.GridSpec(3, 4) # , width_ratios=[0.8, 1, 1, 1])

        ax = fig.add_subplot(gs[0:2, 0])
        import cv2
        # Convert BGR color ordering to RGB.
        image = cv2.imread(
            os.path.join(
                root_dir,
                'figures/motion_prescribed_walking_inverse_model.png'))[
                ..., ::-1]
        ax.imshow(image)
        plt.axis('off')

        muscles = [
            ((0, 0), 'glmax2', 'gluteus maximus', 'Perry', 'GluteusMaximusUpper'),
            ((0, 1), 'iliacus', 'iliacus', 'Perry', 'Iliacus'),
            ((0, 2), 'recfem', 'rectus femoris', '', 'RF'),
            ((1, 0), 'semiten', 'semitendinosus', '', 'MH'),
            ((1, 1), 'bfsh', 'biceps femoris short head', '', 'BF'),
            ((1, 2), 'vaslat', 'vastus lateralis', '', 'VL'),
            ((2, 0), 'gasmed', 'medial gastrocnemius', '', 'GAS'),
            ((2, 1), 'soleus', 'soleus', '', 'SOL'),
            ((2, 2), 'tibant', 'tibialis anterior', '', 'TA'),
        ]
        legend_handles_and_labels = []
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[muscle[0][0], muscle[0][1] + 1])
            # excitation_path = f'/forceset/{muscle[1]}_{self.side}'
            activation_path = f'/forceset/{muscle[1]}_{self.side}/activation'
            legend_musc = []
            if self.inverse:
                inverse_activ = sol_inverse.getStateMat(activation_path)
                # inverse_excit = sol_inverse.getControlMat(excitation_path)
                handle, = self.plot(ax, time_inverse, inverse_activ,
                          label='MocoInverse',
                          color='black',
                          linewidth=2, zorder=2)
                legend_musc.append((handle, 'MocoInverse'))

                # inverse_fine_activ = sol_inverse_fine.getStateMat(activation_path)
                # handle, = self.plot(ax, time_inverse_fine, inverse_fine_activ,
                #                     label='MocoInverse, fine',
                #                     color='gray',
                #                     linestyle=':',
                #                     linewidth=2, zorder=2)
                # legend_musc.append((handle, 'MocoInverse, fine'))
            if self.knee and self.inverse:
                handle, = self.plot(ax, time_inverse_jointreaction,
                                    # sol_inverse_jointreaction.getControlMat(excitation_path),
                          sol_inverse_jointreaction.getStateMat(activation_path),
                          label='MocoInverse, knee',
                          linewidth=2, zorder=3)
                legend_musc.append((handle, 'MocoInverse, knee'))
            if self.inverse:
                if muscle[3] == 'Perry':
                    handle = ax.fill_between(emgPerry['percent_gait_cycle'],
                                    emgPerry[muscle[4]] / 100.0,
                                    np.zeros_like(emgPerry[muscle[4]]),
                                    clip_on=False,
                                    color='lightgray',
                                    label='electromyography')
                else:
                    handle = self.plot(ax, emg['time'],
                                       emg[muscle[4]] * np.max(inverse_activ),
                              shift=False, fill=True, color='lightgray',
                              label='experiment')
                legend_musc.append((handle, 'electromyography'))
            if im == 0:
                legend_handles_and_labels = legend_musc
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
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
                      loc=(0.025, 0.15),
        )
        fig.tight_layout(h_pad=1, pad=0.4)

        self.savefig(fig, os.path.join(root_dir, 'figures/Fig7'))

        # Supplementary CMC comparison.
        # -----------------------------
        if self.cmc:
            sol_cmc = osim.TimeSeriesTable(
                os.path.join(root_dir, 'results',
                             'motion_prescribed_walking_cmc_results',
                             'motion_prescribed_walking_cmc_states.sto'))
            time_cmc = np.array(sol_cmc.getIndependentColumn())
            sol_so = osim.TimeSeriesTable(
                os.path.join(root_dir, 'results',
                             'motion_prescribed_walking_analyze_results',
                             'motion_prescribed_walking_analyze_'
                             'StaticOptimization_activation.sto'))
            time_so = np.array(sol_so.getIndependentColumn())
            # exc_cmc = osim.TimeSeriesTable(
            #     os.path.join(root_dir, 'results',
            #                  'motion_prescribed_walking_cmc_results',
            #                  'motion_prescribed_walking_cmc_controls.sto'))
            # time_exc_cmc = np.array(exc_cmc.getIndependentColumn())

        fig = plt.figure(figsize=(5.2, 3.3))
        gs = gridspec.GridSpec(3, 3)

        legend_handles_and_labels = []
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[muscle[0][0], muscle[0][1]])
            activation_path = f'/forceset/{muscle[1]}_{self.side}/activation'
            legend_musc = []
            if self.inverse:
                inverse_activ = sol_inverse.getStateMat(activation_path)
                handle, = self.plot(ax, time_inverse, inverse_activ,
                                    label='MocoInverse',
                                    color='black',
                                    linewidth=2.5, zorder=2)
                legend_musc.append((handle, 'MocoInverse'))
            if self.cmc:
                so_activ = toarray(
                    sol_so.getDependentColumn(f'{muscle[1]}_{self.side}'))
                handle, = self.plot(ax, time_so, so_activ,
                                    label='Static Optimization',
                                    linewidth=1.5,
                                    zorder=2)
                legend_musc.append((handle, 'Static Optimization'))
                # cmc_excit = toarray(
                #     exc_cmc.getDependentColumn(f'{muscle[1]}_{self.side}'))
                # handle, = self.plot(ax, time_exc_cmc, cmc_excit,
                #                     label='CMC excitation',
                #                     linewidth=2,
                #                     zorder=2)
                # legend_musc.append((handle, 'CMC excitation'))
                cmc_activ = toarray(sol_cmc.getDependentColumn(activation_path))
                handle, = self.plot(ax, time_cmc, cmc_activ,
                                    label='CMC',
                                    linewidth=1.5,
                                    zorder=2)
                legend_musc.append((handle, 'CMC'))
            # if self.inverse:
            #     if muscle[3] == 'Perry':
            #         handle = ax.fill_between(emgPerry['percent_gait_cycle'],
            #                                  emgPerry[muscle[4]] / 100.0,
            #                                  np.zeros_like(emgPerry[muscle[4]]),
            #                                  clip_on=False,
            #                                  color='lightgray',
            #                                  label='electromyography')
            #     else:
            #         handle = self.plot(ax, emg['time'],
            #                            emg[muscle[4]] * np.max(inverse_activ),
            #                            shift=False, fill=True, color='lightgray',
            #                            label='experiment')
            #     legend_musc.append((handle, 'electromyography'))
            if im == 0:
                legend_handles_and_labels = legend_musc
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
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
                      loc=(0.09, 0.47),
                      handlelength=1,
                      fontsize=8,
                      )
        fig.tight_layout(h_pad=1, pad=0.4)

        self.savefig(fig, os.path.join(root_dir, 'figures/S2_Fig'))

        res_to_genforce_labels = dict()
        if self.inverse:
            genforces = utilities.calc_net_generalized_forces(
                model, sol_inverse)
            genforce_labels = genforces.getColumnLabels()
            res_inverse = self.calc_reserves(root_dir, sol_inverse)
            column_labels = res_inverse.getColumnLabels()
            for orig_gen_label in genforce_labels: 
                gen_label = orig_gen_label.replace('_moment', '')
                gen_label = gen_label.replace('_force', '')
                for res_label in column_labels:
                    if gen_label in res_label:
                        res_to_genforce_labels[res_label] = orig_gen_label
            max_res_inverse = -np.inf
            max_res_inverse_percent_genforce = -np.inf
            max_label = ''
            for icol in range(res_inverse.getNumColumns()):
                label = column_labels[icol]
                if (('arm' in label) or 
                        ('elbow' in label) or 
                        ('pro_sup' in label)):
                    continue
                column = utilities.toarray(
                    res_inverse.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                genforce_label = res_to_genforce_labels[label]
                genforce = utilities.toarray(
                    genforces.getDependentColumn(genforce_label))
                max_genforce = np.max(np.abs(genforce))
                max_percent_genforce = 100.0 * (max / max_genforce) 
                if max_percent_genforce > max_res_inverse_percent_genforce:
                    max_res_inverse = max
                    max_label = label
                    max_res_inverse_percent_genforce = max_percent_genforce
                print(f'inverse max abs {label}: {max:.2f} '
                      f'({max_percent_genforce:.2f}% peak generalized force)')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_max_reserve.txt'), 'w') as f:
                f.write(f'{max_res_inverse:.2f} N-m in reserve {max_label}\n')
                f.write(f'{max_res_inverse_percent_genforce:.2f}% of peak '
                        f'generalized force')

        if self.knee and self.inverse:
            genforces = utilities.calc_net_generalized_forces(
                model, sol_inverse_jointreaction)
            genforce_labels = genforces.getColumnLabels()
            res_inverse_jr = self.calc_reserves(root_dir,
                                                sol_inverse_jointreaction)
            column_labels = res_inverse_jr.getColumnLabels()
            max_res_inverse_jr = -np.inf
            max_res_inverse_jr_percent_genforce = -np.inf
            max_label = ''
            for icol in range(res_inverse.getNumColumns()):
                label = column_labels[icol]
                if (('arm' in label) or 
                        ('elbow' in label) or 
                        ('pro_sup' in label)):
                    continue
                column = utilities.toarray(
                    res_inverse_jr.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                genforce_label = res_to_genforce_labels[label]
                genforce = utilities.toarray(
                    genforces.getDependentColumn(genforce_label))
                max_genforce = np.max(np.abs(genforce))
                max_percent_genforce = 100.0 * (max / max_genforce) 
                if max_percent_genforce > max_res_inverse_jr_percent_genforce:
                    max_res_inverse_jr = max
                    max_label = label
                    max_res_inverse_jr_percent_genforce = max_percent_genforce
                print(f'inverse_jr max abs {label}: {max:.2f} '
                      f'({max_percent_genforce:.2f}% peak generalized force)')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_jr_max_reserve.txt'), 'w') as f:
                f.write(f'{max_res_inverse_jr:.2f} N-m in reserve {max_label}\n')
                f.write(f'{max_res_inverse_jr_percent_genforce:.2f}% of peak '
                        f'generalized force')

            maxjr_inverse, avgjr_inverse = \
                self.calc_knee_reaction_force(root_dir, sol_inverse)
            maxjr_inverse_jr, avgjr_inverse_jr = \
                self.calc_knee_reaction_force(root_dir, sol_inverse_jointreaction)
            print(f'Max joint reaction {maxjr_inverse} -> {maxjr_inverse_jr}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_maxjr.txt'), 'w') as f:
                f.write(f'{maxjr_inverse:.1f}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                      'inverse_jr_maxjr.txt'), 'w') as f:
                f.write(f'{maxjr_inverse_jr:.1f}')
            print(f'Average joint reaction {avgjr_inverse} -> {avgjr_inverse_jr}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                                             'inverse_avgjr.txt'), 'w') as f:
                f.write(f'{avgjr_inverse:.1f}')
            with open(os.path.join(root_dir, 'results/motion_prescribed_walking_'
                                             'inverse_jr_avgjr.txt'), 'w') as f:
                f.write(f'{avgjr_inverse_jr:.1f}')

        if self.inverse:
            states = sol_inverse.exportToStatesTrajectory(model)
            duration = sol_inverse.getFinalTime() - sol_inverse.getInitialTime()
            avg_speed = (model.calcMassCenterPosition(states[states.getSize() - 1])[0] -
                         model.calcMassCenterPosition(states[0])[0]) / duration
            print(f'Average speed: {avg_speed}')

        if self.inverse:
            output_fpath = os.path.join(
                root_dir, 'results',
                'motion_prescribed_walking_inverse_solution_report.pdf')
            report = osim.report.Report(
                model,
                self.mocoinverse_solution_file % root_dir,
                output=output_fpath)
            report.generate()

        if self.knee and self.inverse:
            output_fpath = os.path.join(
                root_dir, 'results',
                'motion_prescribed_walking_inverse_jointreaction_solution_'
                'report.pdf')
            report = osim.report.Report(
                model,
                mocoinverse_jr_solution_file % root_dir,
                output=output_fpath)
            report.generate()

    def convergence_metadata(self):
        return [
            {'num_mesh_intervals': 5,
             'solution_file': 'motion_prescribed_walking_solution_5.sto'},
            {'num_mesh_intervals': 10,
             'solution_file': 'motion_prescribed_walking_solution_10.sto'},
            {'num_mesh_intervals': 20,
             'solution_file': 'motion_prescribed_walking_solution_20.sto'},
            {'num_mesh_intervals': 40,
             'solution_file': 'motion_prescribed_walking_solution_40.sto'},
            {'num_mesh_intervals': 90,
             'solution_file': 'motion_prescribed_walking_solution_90.sto'},
            {'num_mesh_intervals': 160,
             'solution_file': 'motion_prescribed_walking_solution_160.sto'},
            {'num_mesh_intervals': 320,
             'solution_file': 'motion_prescribed_walking_solution_320.sto'},
        ]

    def generate_convergence_results(self, root_dir, args):
        # TODO: convergence_tolerance?
        self.parse_args(args)
        if args:
            raise Exception("prescribed-walking: args not valid with "
                            "--convergence.")

        modelProcessor = self.create_model_processor(root_dir)

        duration = self.final_time - self.initial_time
        for md in self.convergence_metadata():
            num_mesh_intervals = md['num_mesh_intervals']
            print(f'Convergence analysis: using {num_mesh_intervals} mesh '
                  'intervals.')
            if 'generate' in md and not md['generate']:
                print(f'Skipping {num_mesh_intervals}.')
                continue
            # TODO: We could still get off-by-one mesh intervals.
            mesh_interval = duration / num_mesh_intervals
            solution_filepath = os.path.join(root_dir, 'results',
                                              'convergence',
                                              md['solution_file'])
            self.solve_inverse(root_dir, modelProcessor, mesh_interval,
                               solution_filepath)

            # TODO
            # sol_inverse_fine = osim.MocoTrajectory(
            #     self.mocoinverse_fine_solution_file % root_dir)
            # time_inverse_fine = sol_inverse_fine.getTimeMat()
            # most_neg = self.calc_negative_muscle_forces_base(model,
            #                                                  sol_inverse_fine)
            # if most_neg < -0.005:
            #     raise Exception("Muscle forces are too negative! "
            #                     "sol_inverse_fine")
