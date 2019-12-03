import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import opensim as osim

from moco_paper_result import MocoPaperResult

class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        self.initial_time = 0.81
        self.half_time = 1.385
        self.final_time = 1.96
        self.mesh_interval = 0.035
        self.inverse_solution_relpath = \
            'results/motion_tracking_walking_inverse_solution.sto'
        self.tracking_solution_relpath_prefix = \
            'results/motion_tracking_walking_solution'
        self.tracking_weights = [1,  1, 0.001]
        self.effort_weights =   [0.001, 1, 1]
        self.cmap = 'viridis'
        self.cmap_indices = [0.1, 0.5, 0.9]

    def create_model_processor(self, root_dir, for_inverse=False):

        model = osim.Model()
        if for_inverse:
            model = osim.Model(os.path.join(root_dir,
                'resources/Rajagopal2016/'
                'subject_walk_armless_lumbar_80musc.osim'))
        else:
            model = osim.Model(os.path.join(root_dir,
                'resources/Rajagopal2016/'
                'subject_walk_armless_contact_bounded_lumbar_80musc.osim'))

        def add_reserve(model, coord, optimal_force, max_control):
            actu = osim.ActivationCoordinateActuator()
            actu.set_coordinate(coord)
            if coord.startswith('pelvis'):
                prefix = 'residual_'
            else:
                prefix = 'reserve_'
            actu.setName(prefix + coord)
            actu.setOptimalForce(optimal_force)
            actu.setMinControl(-max_control)
            actu.setMaxControl(max_control)
            model.addForce(actu)
        reserves_max = 250 if for_inverse else 1
        add_reserve(model, 'lumbar_extension', 1, reserves_max)
        add_reserve(model, 'lumbar_bending', 1, reserves_max)
        add_reserve(model, 'lumbar_rotation', 1, reserves_max)
        add_reserve(model, 'pelvis_tilt', 1, reserves_max)
        add_reserve(model, 'pelvis_list', 1, reserves_max)
        add_reserve(model, 'pelvis_rotation', 1, reserves_max)
        add_reserve(model, 'pelvis_tx', 1, reserves_max)
        add_reserve(model, 'pelvis_ty', 1, reserves_max)
        add_reserve(model, 'pelvis_tz', 1, reserves_max)
        add_reserve(model, 'hip_flexion_r', 0.1, reserves_max)
        add_reserve(model, 'hip_adduction_r', 0.1, reserves_max)
        add_reserve(model, 'hip_rotation_r', 0.1, reserves_max)
        add_reserve(model, 'knee_angle_r', 0.1, reserves_max)
        add_reserve(model, 'ankle_angle_r', 0.1, reserves_max)
        add_reserve(model, 'hip_flexion_l', 0.1, reserves_max)
        add_reserve(model, 'hip_adduction_l', 0.1, reserves_max)
        add_reserve(model, 'hip_rotation_l', 0.1, reserves_max)
        add_reserve(model, 'knee_angle_l', 0.1, reserves_max)
        add_reserve(model, 'ankle_angle_l', 0.1, reserves_max)

        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        if for_inverse:
            modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
            ext_loads_xml = os.path.join(root_dir,
                    'resources/Rajagopal2016/grf_walk.xml')
            modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))

        return modelProcessor

    def get_solution_path(self, root_dir, tracking_weight, effort_weight):
        trackingWeight = str(tracking_weight).replace('.','_')
        effortWeight = str(effort_weight).replace('.','_')
        return os.path.join(root_dir, 
                    f'{self.tracking_solution_relpath_prefix}'
                    f'_trackingWeight{trackingWeight}'
                    f'_effortWeight{effortWeight}.sto')

    def get_solution_path_fullcycle(self, root_dir, tracking_weight,
            effort_weight):
        trackingWeight = str(tracking_weight).replace('.','_')
        effortWeight = str(effort_weight).replace('.','_')
        return os.path.join(root_dir, 
                    f'{self.tracking_solution_relpath_prefix}'
                    f'_trackingWeight{trackingWeight}'
                    f'_effortWeight{effortWeight}_fullcycle.sto')

    def get_solution_path_grfs(self, root_dir, tracking_weight,
            effort_weight):
        trackingWeight = str(tracking_weight).replace('.','_')
        effortWeight = str(effort_weight).replace('.','_')
        return os.path.join(root_dir, 
                    f'{self.tracking_solution_relpath_prefix}'
                    f'_trackingWeight{trackingWeight}'
                    f'_effortWeight{effortWeight}_fullcycle_grfs.sto')

    def load_table(self, table_path):
        num_header_rows = 1
        with open(table_path) as f:
            for line in f:
                if not line.startswith('endheader'):
                    num_header_rows += 1
                else:
                    break
        return np.genfromtxt(table_path, names=True, delimiter='\t',
                                 skip_header=num_header_rows)


    def run_inverse_problem(self, root_dir):

        modelProcessor = self.create_model_processor(root_dir, 
            for_inverse=True)

        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor);
        tableProcessor = osim.TableProcessor(os.path.join(root_dir,
                'resources/Rajagopal2016/coordinates.mot'))
        tableProcessor.append(osim.TabOpLowPassFilter(6))
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        inverse.setKinematics(tableProcessor)
        inverse.set_kinematics_allow_extra_columns(True);
        inverse.set_initial_time(self.initial_time);
        inverse.set_final_time(self.half_time);
        inverse.set_mesh_interval(self.mesh_interval);

        solution = inverse.solve()
        solution.getMocoSolution().write(
            os.path.join(root_dir, self.inverse_solution_relpath))

    def run_tracking_problem(self, root_dir, previous_solution, 
            tracking_weight=1, effort_weight=1):

        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        model.initSystem()

        # Count the number of Force objects in the model. We'll use this to 
        # normalize the control effort cost.
        numForces = 0
        for actu in model.getComponentsList():
            if (actu.getConcreteClassName().endswith('Muscle') or 
                actu.getConcreteClassName().endswith('Actuator')):
                numForces += 1

        # Construct the base tracking problem
        # -----------------------------------
        track = osim.MocoTrack()
        track.setName('tracking_walking')
        track.setModel(modelProcessor);
        tableProcessor = osim.TableProcessor(os.path.join(root_dir,
                'resources/Rajagopal2016/coordinates.mot'))
        tableProcessor.append(osim.TabOpLowPassFilter(6))
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        track.setStatesReference(tableProcessor)
        track.set_allow_unused_references(True);
        track.set_track_reference_position_derivatives(True);
        # track.set_scale_state_weights_with_range(True);
        if previous_solution.empty():
            track.set_apply_tracked_states_to_guess(True);
        track.set_states_global_tracking_weight(
                tracking_weight / (2 * model.getNumCoordinates()))
        track.set_control_effort_weight(20 * effort_weight / numForces);
        track.set_initial_time(self.initial_time);
        track.set_final_time(self.half_time);
        track.set_mesh_interval(self.mesh_interval);

        # Customize the base tracking problem
        # -----------------------------------
        study = track.initialize()
        problem = study.updProblem()
        problem.setTimeBounds(self.initial_time, 
                [self.half_time-0.2, self.half_time+0.2])
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_tx/value', [], 
                0.446)
        # Set the initial values for the lumbar coordinates to avoid 
        # weird torso movements. 
        problem.setStateInfo('/jointset/back/lumbar_extension/value', [], -0.12)
        problem.setStateInfo('/jointset/back/lumbar_bending/value', [], 0)
        problem.setStateInfo('/jointset/back/lumbar_rotation/value', [], 0.04)
        # Update the control effort goal to a cost of transport type cost.
        effort = osim.MocoControlGoal().safeDownCast(
                problem.updGoal('control_effort'))
        effort.setDivideByDisplacement(True)

        speedGoal = osim.MocoAverageSpeedGoal('speed')
        speedGoal.set_desired_average_speed(1.235)
        problem.addGoal(speedGoal)

        distanceConstraint = osim.MocoMinimumDistanceConstraint()
        distanceConstraint.setName('distance_constraint')
        distance = 0.15
        distanceConstraint.addFramePair(osim.MocoMinimumDistanceConstraintPair(
            '/bodyset/calcn_l', '/bodyset/calcn_r', distance))
        distanceConstraint.addFramePair(osim.MocoMinimumDistanceConstraintPair(
            '/bodyset/toes_l', '/bodyset/toes_r', distance))
        distanceConstraint.addFramePair(osim.MocoMinimumDistanceConstraintPair(
            '/bodyset/calcn_l', '/bodyset/toes_r', distance))
        distanceConstraint.addFramePair(osim.MocoMinimumDistanceConstraintPair(
            '/bodyset/toes_l', '/bodyset/calcn_r', distance))
        problem.addPathConstraint(distanceConstraint)

        # Symmetry contraints
        # -------------------
        statesRef = osim.TimeSeriesTable('tracking_walking_tracked_states.sto')
        initIndex = statesRef.getNearestRowIndexForTime(self.initial_time)
        symmetry = osim.MocoPeriodicityGoal('symmetry')
        # Symmetric coordinate values (except for pelvis_tx) and speeds.
        for coord in model.getComponentsList():
            if not type(coord) is osim.Coordinate: continue

            if coord.getName().endswith('_r'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0),
                    coord.getStateVariableNames().get(0).replace('_r/', '_l/')))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1),
                    coord.getStateVariableNames().get(1).replace('_r/', '_l/')))
            elif coord.getName().endswith('_l'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0),
                    coord.getStateVariableNames().get(0).replace('_l/', '_r/')))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1),
                    coord.getStateVariableNames().get(1).replace('_l/', '_r/')))
            elif (coord.getName().endswith('_bending') or 
                  coord.getName().endswith('_rotation') or 
                  coord.getName().endswith('_tz') or
                  coord.getName().endswith('_list')): 
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0)))
                symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1)))
            elif not coord.getName().endswith('_tx'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(0)))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coord.getStateVariableNames().get(1)))
        symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
            '/jointset/ground_pelvis/pelvis_tx/speed'))
        # Symmetric activations.
        for actu in model.getComponentsList():
            if (not actu.getConcreteClassName().endswith('Muscle') and 
                not actu.getConcreteClassName().endswith('Actuator')): continue
            if actu.getName().endswith('_r'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    actu.getStateVariableNames().get(0),
                    actu.getStateVariableNames().get(0).replace('_r/', '_l/')))
            elif actu.getName().endswith('_l'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    actu.getStateVariableNames().get(0),
                    actu.getStateVariableNames().get(0).replace('_l/', '_r/')))
            elif (actu.getName().endswith('_bending') or 
                  actu.getName().endswith('_rotation') or
                  actu.getName().endswith('_tz') or
                  actu.getName().endswith('_list')):
                symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                    actu.getStateVariableNames().get(0),
                    actu.getStateVariableNames().get(0)))
            else:
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    actu.getStateVariableNames().get(0),
                    actu.getStateVariableNames().get(0)))
        problem.addGoal(symmetry)

        # Configure the solver
        # --------------------
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        solver.resetProblem(problem)
        solver.set_optim_constraint_tolerance(1e-3)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_multibody_dynamics_mode('implicit')
        solver.set_minimize_implicit_multibody_accelerations(True)
        solver.set_implicit_multibody_accelerations_weight(
            0.0001 / model.getNumCoordinates())
        solver.set_implicit_multibody_acceleration_bounds(
                osim.MocoBounds(-200, 200))

        # Set the guess
        # -------------
        # If the previous solution argument is empty, use the inverse solution
        # to construct a guess.
        guess = osim.MocoTrajectory()
        if previous_solution.empty():
            # Create a guess compatiable with this problem.
            guess = solver.createGuess()
            # Load the inverse problem solution and set its states and controls 
            # trajectories to the guess.
            inverseSolution = osim.MocoTrajectory(
                os.path.join(root_dir, self.inverse_solution_relpath))
            inverseStatesTable = inverseSolution.exportToStatesTable()
            guess.insertStatesTrajectory(inverseStatesTable, True)
            inverseControlsTable = inverseSolution.exportToControlsTable()
            guess.insertControlsTrajectory(inverseControlsTable, True)
        else:
            guess = previous_solution
        solver.setGuess(guess)

        # Solve and print solution.
        # -------------------------
        solution = study.solve()
        trackingWeight = str(tracking_weight).replace('.','_')
        effortWeight = str(effort_weight).replace('.','_')
        solution.write(os.path.join(root_dir, 
            f'{self.tracking_solution_relpath_prefix}'
            f'_trackingWeight{trackingWeight}_effortWeight{effortWeight}.sto'))
        # Create a full gait cycle trajectory from the periodic solution.
        fullTraj = osim.createPeriodicTrajectory(solution)
        fullTraj.write(os.path.join(root_dir, 
            f'{self.tracking_solution_relpath_prefix}'
            f'_trackingWeight{trackingWeight}'
            f'_effortWeight{effortWeight}_fullcycle.sto'))

        # Compute ground reaction forces generated by contact sphere from the 
        # full gait cycle trajectory.
        forceNamesRightFoot = ['contactSphereHeel_r',
                'contactLateralRearfoot_r', 'contactLateralMidfoot_r',
                'contactLateralToe_r', 'contactMedialToe_r',
                'contactMedialMidfoot_r']
        forceNamesLeftFoot = ['contactSphereHeel_l',
                'contactLateralRearfoot_l', 'contactLateralMidfoot_l',
                'contactLateralToe_l', 'contactMedialToe_l',
                'contactMedialMidfoot_l']
        externalLoads = osim.createExternalLoadsTableForGait(
                model, fullTraj, forceNamesRightFoot, forceNamesLeftFoot)
        osim.writeTableToFile(externalLoads, os.path.join(root_dir, 
                f'{self.tracking_solution_relpath_prefix}'
                f'_trackingWeight{trackingWeight}'
                f'_effortWeight{effortWeight}_fullcycle_grfs.sto'))

        # study.visualize(fullTraj)

        # solution = osim.MocoTrajectory(
        #     self.get_solution_path_fullcycle(root_dir, tracking_weight,
        #           effort_weight))
        # study.visualize(solution)

        return solution

    def generate_results(self, root_dir, args):

        # Run inverse problem to generate first initial guess.
        self.run_inverse_problem(root_dir)

        # Run tracking problem, sweeping across different effort weights.
        solution = osim.MocoTrajectory()
        weights = zip(self.tracking_weights, self.effort_weights)
        for tracking_weight, effort_weight in weights:
            solution = self.run_tracking_problem(root_dir, solution, 
                    tracking_weight=tracking_weight,
                    effort_weight=effort_weight)

    def report_results(self, root_dir, args):

        iterate = zip(self.tracking_weights, self.effort_weights, 
                self.cmap_indices)
        
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(3, 2)
        ax_grf_x = fig.add_subplot(gs[0,0])
        ax_grf_y = fig.add_subplot(gs[1,0])
        ax_grf_z = fig.add_subplot(gs[2,0])
        ax_hip = fig.add_subplot(gs[0,1])
        ax_knee = fig.add_subplot(gs[1,1])
        ax_ankle = fig.add_subplot(gs[2,1])
        cmap = cm.get_cmap(self.cmap)
        for tracking_weight, effort_weight, cmap_index in iterate:
            color = cmap(cmap_index)
            full_path = self.get_solution_path_fullcycle(root_dir, 
                    tracking_weight, effort_weight)
            full_table = osim.MocoTrajectory(full_path)
            grf_path = self.get_solution_path_grfs(root_dir, 
                    tracking_weight, effort_weight)
            grf_table = self.load_table(grf_path)

            ax_grf_x.plot(grf_table['time'], 
                          grf_table['ground_force_l_vx'],
                    color=color)

        fig.tight_layout()
        fig.savefig(os.path.join(root_dir, 
                'figures/motion_tracking_walking.png'))


        with open(os.path.join(root_dir, 'results/'
                'motion_tracking_walking_durations.txt'), 'w') as f:
            for tracking_weight, effort_weight, cmap_index in iterate:
                print('cmap_index', cmap_index)
                sol_path = self.get_solution_path(root_dir, tracking_weight,
                        effort_weight)
                trackingWeight = str(tracking_weight).replace('.','_')
                effortWeight = str(effort_weight).replace('.','_')
                sol_table = osim.TimeSeriesTable(sol_path)
                duration = sol_table.getTableMetaDataString('solver_duration')
                duration = float(duration) / 60.0 / 60.0
                print(f'duration (track={trackingWeight}, '
                      f'effort={effortWeight}): ', duration)              
                f.write(f'(track={trackingWeight}, effort={effortWeight}): '
                        f'{duration:.2f}\n')



        # sol_track = osim.MocoTrajectory(self.mocotrack_solution_file % root_dir)
        # time_track = sol_track.getTimeMat()


        # effort_weight = 0.1
        # effortWeight = str(self.effort_weights[0]).replace('.','_')
        # solutionPath = os.path.join(root_dir, 
        #         f'{self.tracking_solution_relpath_prefix}'
        #         f'_effortWeight{effortWeight}.sto')

        # solution = osim.MocoTrajectory(solutionPath)

        # fullTraj = osim.createPeriodicTrajectory(solution)


        # fullTrajPath = os.path.join(root_dir, 
        #     f'{self.tracking_solution_relpath_prefix}'
        #     f'_effortWeight{effortWeight}_fullcycle.sto')
        # fullTraj.write(fullTrajPath)


        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()

        trajectory_filepath = self.get_solution_path_fullcycle(root_dir, 
                self.tracking_weights[-1], 
                self.effort_weights[-1])

        ref_files = list()
        ref_files.append('tracking_walking_tracked_states.sto')
        for tracking_weight, effort_weight, cmap_index in iterate:
            ref_files.append(self.get_solution_path_fullcycle(root_dir,
                tracking_weight, effort_weight))

        report = osim.report.Report(model=model, 
                trajectory_filepath=trajectory_filepath, 
                ref_files=ref_files, bilateral=False)
        report.generate()



