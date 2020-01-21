import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities

# TODO: try different foot spacing
# TODO: fix oscillations in GRFz.
# TODO: semimem and gasmed forces are negative.
# TODO: feet are crossing over too much (b/c adductor passive force?)
# TODO: remove reserves from tracking problem?

class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        super(MotionTrackingWalking, self).__init__()
        self.initial_time = 0.81
        self.half_time = 1.385
        self.final_time = 1.96
        self.mesh_interval = 0.035
        self.inverse_solution_relpath = \
            'results/motion_tracking_walking_inverse_solution.sto'
        self.tracking_solution_relpath_prefix = \
            'results/motion_tracking_walking_solution'
        self.tracking_weights = [1, 1, 0.01]
        self.effort_weights = [0.1, 10, 10]
        self.cmap = 'nipy_spectral'
        self.cmap_indices = [0.2, 0.5, 0.9]
        self.legend_entries = ['track', 'track\n+\neffort', 'effort']
        self.legend_kwargs = [{'horizontalalignment': 'left',
                               'verticalalignment': 'middle',
                               'bbox': {'boxstyle': 'square,pad=1'}},
                              {'horizontalalignment': 'left',
                               'verticalalignment': 'middle',
                               'bbox': {'boxstyle': 'square,pad=1'}},
                              {'horizontalalignment': 'center',
                               'verticalalignment': 'bottom',
                               'bbox': {'boxstyle': 'square,pad=1'}},
                              ]

    def create_model_processor(self, root_dir, for_inverse=False):
        
        model = osim.Model(os.path.join(root_dir,
                'resources/Rajagopal2016/'
                'subject_walk_armless_contact_bounded_80musc.osim'))

        if for_inverse:
            forceSet = model.getForceSet()
            numContacts = 0
            for i in np.arange(forceSet.getSize()):
                forceName = forceSet.get(int(i)).getName()
                if 'contact' in forceName: numContacts += 1

            contactsRemoved = 0
            while contactsRemoved < numContacts:
                for i in np.arange(forceSet.getSize()):
                    forceName = forceSet.get(int(i)).getName()
                    if 'contact' in forceName: 
                        print('Force removed: ', forceSet.get(int(i)).getName())
                        forceSet.remove(int(i))
                        contactsRemoved += 1
                        break

        def add_reserve(model, coord, optimal_force, max_control):
            actu = osim.ActivationCoordinateActuator()
            actu.set_coordinate(coord)
            if coord.startswith('pelvis'):
                prefix = 'residual_'
            elif coord.startswith('lumbar'):
                prefix = 'torque_'
            else:
                prefix = 'reserve_'
            actu.setName(prefix + coord)
            actu.setOptimalForce(optimal_force)
            actu.setMinControl(-max_control)
            actu.setMaxControl(max_control)
            model.addForce(actu)
        reserves_max = 250 if for_inverse else 1
        add_reserve(model, 'lumbar_extension', 1, 50)
        add_reserve(model, 'lumbar_bending', 1, 50)
        add_reserve(model, 'lumbar_rotation', 1, 50)
        add_reserve(model, 'pelvis_tilt', 1, reserves_max)
        add_reserve(model, 'pelvis_list', 1, reserves_max)
        add_reserve(model, 'pelvis_rotation', 1, reserves_max)
        add_reserve(model, 'pelvis_tx', 1, reserves_max)
        add_reserve(model, 'pelvis_ty', 1, reserves_max)
        add_reserve(model, 'pelvis_tz', 1, reserves_max)
        optimal_force = 0.1
        add_reserve(model, 'hip_flexion_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_adduction_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_rotation_r', optimal_force, reserves_max)
        add_reserve(model, 'knee_angle_r', optimal_force, reserves_max)
        add_reserve(model, 'ankle_angle_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_flexion_l', optimal_force, reserves_max)
        add_reserve(model, 'hip_adduction_l', optimal_force, reserves_max)
        add_reserve(model, 'hip_rotation_l', optimal_force, reserves_max)
        add_reserve(model, 'knee_angle_l', optimal_force, reserves_max)
        add_reserve(model, 'ankle_angle_l', optimal_force, reserves_max)

        modelProcessor = osim.ModelProcessor(model)

        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        if for_inverse:
            ext_loads_xml = os.path.join(root_dir,
                    'resources/Rajagopal2016/grf_walk.xml')
            modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))

        model = modelProcessor.process()
        model.initSystem()
        muscles = model.updMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = muscles.get(int(imusc))
            if 'gas' in muscle.getName() or 'soleus' in muscle.getName():
                muscle.set_ignore_tendon_compliance(False)
                muscle.set_max_isometric_force(0.25*muscle.get_max_isometric_force())

        modelProcessorTendonCompliance = osim.ModelProcessor(model)
        modelProcessorTendonCompliance.append(
                osim.ModOpUseImplicitTendonComplianceDynamicsDGF())

        return modelProcessorTendonCompliance

    def get_solution_path(self, root_dir, tracking_weight, effort_weight):
        trackingWeight = str(tracking_weight).replace('.', '_')
        effortWeight = str(effort_weight).replace('.', '_')
        return os.path.join(root_dir, 
                    f'{self.tracking_solution_relpath_prefix}'
                    f'_trackingWeight{trackingWeight}'
                    f'_effortWeight{effortWeight}.sto')

    def get_solution_path_fullcycle(self, root_dir, tracking_weight,
            effort_weight):
        trackingWeight = str(tracking_weight).replace('.', '_')
        effortWeight = str(effort_weight).replace('.', '_')
        return os.path.join(root_dir, 
                    f'{self.tracking_solution_relpath_prefix}'
                    f'_trackingWeight{trackingWeight}'
                    f'_effortWeight{effortWeight}_fullcycle.sto')

    def get_solution_path_grfs(self, root_dir, tracking_weight,
            effort_weight):
        trackingWeight = str(tracking_weight).replace('.', '_')
        effortWeight = str(effort_weight).replace('.', '_')
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

    def calc_reserves(self, root_dir, solution):
        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        output = osim.analyze(model, solution, ['.*reserve.*actuation'])
        return output

    def calc_muscle_mechanics(self, root_dir, solution):
        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        output = osim.analyze(model, solution,
                              ['.*gasmed.*\|tendon_force',
                               '.*gasmed.*\|normalized_fiber_length',
                               '.*semimem.*\|tendon_force',
                               '.*semimem.*\|normalized_fiber_length'])
        return output

    def run_inverse_problem(self, root_dir):

        modelProcessor = self.create_model_processor(root_dir,
                                                     for_inverse=True)

        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        tableProcessor = osim.TableProcessor(os.path.join(root_dir,
                'resources/Rajagopal2016/coordinates.mot'))
        tableProcessor.append(osim.TabOpLowPassFilter(6))
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        inverse.setKinematics(tableProcessor)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.half_time)
        inverse.set_mesh_interval(self.mesh_interval)

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
        track.setModel(modelProcessor)
        
        if self.coordinate_tracking:
            tableProcessor = osim.TableProcessor(os.path.join(root_dir,
                    'resources/Rajagopal2016/coordinates.mot'))
            tableProcessor.append(osim.TabOpLowPassFilter(6))
            tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
            track.setStatesReference(tableProcessor)
            track.set_states_global_tracking_weight(
                tracking_weight / (2 * model.getNumCoordinates()))
            # Don't track some pelvis coordinates to avoid poor walking motion
            # solutions.
            stateWeights = osim.MocoWeightSet()
            weightList = list()
            weightList.append(('/jointset/ground_pelvis/pelvis_tz/value', 0))
            weightList.append(('/jointset/ground_pelvis/pelvis_list/value', 0))
            weightList.append(('/jointset/ground_pelvis/pelvis_tilt/value', 0))
            weightList.append(('/jointset/ground_pelvis/pelvis_rotation/value', 0))
            for weight in weightList:
                stateWeights.cloneAndAppend(osim.MocoWeight(weight[0], weight[1]))
            track.set_states_weight_set(stateWeights)
            if previous_solution.empty():
                track.set_apply_tracked_states_to_guess(True)
                # track.set_scale_state_weights_with_range(True);
        else:
            track.setMarkersReferenceFromTRC(os.path.join(root_dir,
                    'resources/Rajagopal2016/markers.trc'))
            track.set_markers_global_tracking_weight(
                tracking_weight / (2 * model.getNumMarkers()))
            iktool = osim.InverseKinematicsTool(os.path.join(root_dir,
                    'resources/Rajagopal2016/ik_setup_walk.xml'))
            iktasks = iktool.getIKTaskSet()
            markerWeights = osim.MocoWeightSet()
            for marker in model.getComponentsList():
                if not type(marker) is osim.Marker: continue
                for i in np.arange(iktasks.getSize()):
                    iktask = iktasks.get(int(i))
                    if iktask.getName() == marker.getName():
                        weight = osim.MocoWeight(iktask.getName(), 
                            iktask.getWeight())
                        markerWeights.cloneAndAppend(weight)
            track.set_markers_weight_set(markerWeights)

        track.set_allow_unused_references(True)
        track.set_track_reference_position_derivatives(True)
        track.set_control_effort_weight(effort_weight / numForces)
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.half_time)
        track.set_mesh_interval(self.mesh_interval)

        # Customize the base tracking problem
        # -----------------------------------
        study = track.initialize()
        problem = study.updProblem()
        problem.setTimeBounds(self.initial_time, 
                [self.half_time-0.2, self.half_time+0.2])
        
        # Set the initial values for the lumbar and pelvis coordinates that 
        # produce "normal" walking motions.
        problem.setStateInfo('/jointset/back/lumbar_extension/value', [], -0.12)
        problem.setStateInfo('/jointset/back/lumbar_bending/value', [], 0)
        problem.setStateInfo('/jointset/back/lumbar_rotation/value', [], 0.04)
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_tx/value', [], 0.446)
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_tilt/value', [], 0) 
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_list/value', [], 0)
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_rotation/value', [], 0)

        # Update the control effort goal to a cost of transport type cost.
        effort = osim.MocoControlGoal().safeDownCast(
                problem.updGoal('control_effort'))
        effort.setDivideByDisplacement(True)
        # Weight residual and reserve actuators low in the effort cost since they
        # are already weak.    
        if effort_weight:
            for actu in model.getComponentsList():
                if actu.getConcreteClassName().endswith('Actuator'):
                    effort.setWeightForControl(actu.getAbsolutePathString(), 
                        0.001)

        speedGoal = osim.MocoAverageSpeedGoal('speed')
        speedGoal.set_desired_average_speed(1.235)
        problem.addGoal(speedGoal)

        # MocoFrameDistanceConstraint
        # ---------------------------
        if self.coordinate_tracking:
            distanceConstraint = osim.MocoFrameDistanceConstraint()
            distanceConstraint.setName('distance_constraint')
            distance = 0.15
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/calcn_l', '/bodyset/calcn_r', distance, np.inf))
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/toes_l', '/bodyset/toes_r', distance, np.inf))
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/calcn_l', '/bodyset/toes_r', distance, np.inf))
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/toes_l', '/bodyset/calcn_r', distance, np.inf))
            problem.addPathConstraint(distanceConstraint)

        # Symmetry constraints
        # --------------------
        statesRef = osim.TimeSeriesTable('tracking_walking_tracked_states.sto')
        initIndex = statesRef.getNearestRowIndexForTime(self.initial_time)
        symmetry = osim.MocoPeriodicityGoal('symmetry')
        # Symmetric coordinate values (except for pelvis_tx) and speeds.
        for coord in model.getComponentsList():
            if not type(coord) is osim.Coordinate: continue

            coordName = coord.getName()
            coordValue = coord.getStateVariableNames().get(0)
            coordSpeed = coord.getStateVariableNames().get(1)

            if coordName.endswith('_r'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordValue, coordValue.replace('_r/', '_l/')))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordSpeed, coordSpeed.replace('_r/', '_l/')))
            elif coordName.endswith('_l'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordValue, coordValue.replace('_l/', '_r/')))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordSpeed, coordSpeed.replace('_l/', '_r/')))
            elif (coordName.endswith('_bending') or
                  coordName.endswith('_rotation') or
                  coordName.endswith('_tz') or
                  coordName.endswith('_list')):
                # This does not include hip rotation,
                # because that ends with _l or _r.
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordValue))
                symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                    coordSpeed))
            elif not coordName.endswith('_tx'):
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordValue))
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordSpeed))
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

        # Contact tracking
        # ----------------
        forceNamesRightFoot = ['forceset/contactSphereHeel_r',
                               'forceset/contactLateralRearfoot_r',
                               'forceset/contactLateralMidfoot_r',
                               'forceset/contactLateralToe_r',
                               'forceset/contactMedialToe_r',
                               'forceset/contactMedialMidfoot_r']
        forceNamesLeftFoot = ['forceset/contactSphereHeel_l',
                              'forceset/contactLateralRearfoot_l',
                              'forceset/contactLateralMidfoot_l',
                              'forceset/contactLateralToe_l',
                              'forceset/contactMedialToe_l',
                              'forceset/contactMedialMidfoot_l']
        if self.contact_tracking:
            contactTracking = osim.MocoContactTrackingGoal('contact', 0.001)
            contactTracking.setExternalLoadsFile(
                'resources/Rajagopal2016/grf_walk.xml')
            contactTracking.addContactGroup(forceNamesRightFoot, 'Right_GRF')
            contactTracking.addContactGroup(forceNamesLeftFoot, 'Left_GRF')
            contactTracking.setProjection("plane")
            contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))
            problem.addGoal(contactTracking)

        # Configure the solver
        # --------------------
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        solver.resetProblem(problem)
        solver.set_optim_constraint_tolerance(1e-3)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_multibody_dynamics_mode('implicit')
        solver.set_minimize_implicit_multibody_accelerations(True)
        solver.set_implicit_multibody_accelerations_weight(
            1e-6 / model.getNumCoordinates())
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
        trackingWeight = str(tracking_weight).replace('.', '_')
        effortWeight = str(effort_weight).replace('.', '_')
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
        externalLoads = osim.createExternalLoadsTableForGait(
                model, fullTraj, forceNamesRightFoot, forceNamesLeftFoot)
        osim.writeTableToFile(externalLoads, os.path.join(root_dir, 
                f'{self.tracking_solution_relpath_prefix}'
                f'_trackingWeight{trackingWeight}'
                f'_effortWeight{effortWeight}_fullcycle_grfs.sto'))

        # study.visualize(fullTraj)

        return solution

    def parse_args(self, args):
        self.skip_inverse = False
        self.coordinate_tracking = False
        self.contact_tracking = False
        if len(args) == 0: return
        print('Received arguments {}'.format(args))
        if 'skip-inverse' in args:
            self.skip_inverse = True
        if 'coordinate-tracking' in args:
            self.coordinate_tracking = True
        if 'contact-tracking' in args:
            self.contact_tracking = True

    def generate_results(self, root_dir, args):
        self.parse_args(args)

        # Run inverse problem to generate first initial guess.
        if not self.skip_inverse:
            self.run_inverse_problem(root_dir)

        # Run tracking problem, sweeping across different effort weights.
        solution = osim.MocoTrajectory()
        # solution = osim.MocoTrajectory(self.get_solution_path(root_dir,
            # self.tracking_weights[0], self.effort_weights[0]))
        weights = zip(self.tracking_weights, self.effort_weights)
        for tracking_weight, effort_weight in weights:
            solution = self.run_tracking_problem(root_dir, solution, 
                    tracking_weight=tracking_weight,
                    effort_weight=effort_weight)

    def report_results(self, root_dir, args):
        self.parse_args(args)

        modelProcessor = self.create_model_processor(root_dir)
        model = modelProcessor.process()
        state = model.initSystem()
        mass = model.getTotalMass(state)
        gravity = model.getGravity()
        BW = mass*abs(gravity[1])

        for tracking_weight, effort_weight, cmap_index in zip(
                self.tracking_weights, self.effort_weights,
                self.cmap_indices):
            solution = osim.MocoTrajectory(
                self.get_solution_path_fullcycle(root_dir, tracking_weight,
                      effort_weight))
            osim.visualize(model, solution.exportToStatesTable())

        emg = self.load_electromyography(root_dir)

        fig = plt.figure(figsize=(7.5, 7))
        gs = gridspec.GridSpec(36, 3)
        ax_time = fig.add_subplot(gs[0:12, 0])
        ax_grf_x = fig.add_subplot(gs[12:20, 0])
        ax_grf_y = fig.add_subplot(gs[20:28, 0])
        ax_grf_z = fig.add_subplot(gs[28:36, 0])
        ax_add = fig.add_subplot(gs[0:9, 1])
        ax_hip = fig.add_subplot(gs[9:18, 1])
        ax_knee = fig.add_subplot(gs[18:27, 1])
        ax_ankle = fig.add_subplot(gs[27:36, 1])
        ax_list = list()
        ax_list.append(ax_grf_x)
        ax_list.append(ax_grf_y)
        ax_list.append(ax_grf_z)
        ax_list.append(ax_add)
        ax_list.append(ax_hip)
        ax_list.append(ax_knee)
        ax_list.append(ax_ankle)
        muscles = [
            (fig.add_subplot(gs[0:4, 2]), 'glmax2', 'gluteus maximus', 'GMAX'),
            (fig.add_subplot(gs[4:8, 2]), 'psoas', 'psoas', 'PSOAS'),
            (fig.add_subplot(gs[8:12, 2]), 'semiten', 'semitendinosus', 'MH'),
            (fig.add_subplot(gs[12:16, 2]), 'recfem', 'rectus femoris', 'RF'),
            (fig.add_subplot(gs[16:20, 2]), 'bfsh', 'biceps femoris short head', 'BF'),
            (fig.add_subplot(gs[20:24, 2]), 'vaslat', 'vastus lateralis', 'VL'),
            (fig.add_subplot(gs[24:28, 2]), 'gasmed', 'medial gastrocnemius', 'GAS'),
            (fig.add_subplot(gs[28:32, 2]), 'soleus', 'soleus', 'SOL'),
            (fig.add_subplot(gs[32:36, 2]), 'tibant', 'tibialis anterior', 'TA'),
        ]
        cmap = cm.get_cmap(self.cmap)
        title_fs = 10
        lw = 2.5

        # experimental stride time
        # ax_time.bar(0, self.final_time - self.initial_time, color='black')

        # experimental ground reactions
        grf_table = self.load_table(os.path.join(root_dir,
                'resources', 'Rajagopal2016', 'grf_walk.mot'))
        grf_start = np.argmin(abs(grf_table['time']-self.initial_time))
        grf_end = np.argmin(abs(grf_table['time']-self.final_time))

        time_grfs = grf_table['time'][grf_start:grf_end]
        pgc_grfs = np.linspace(0, 100, len(time_grfs))
        ax_grf_x.plot(pgc_grfs, 
            grf_table['ground_force_l_vx'][grf_start:grf_end]/BW, 
            color='black', lw=lw+1.0)
        ax_grf_y.plot(pgc_grfs, 
            grf_table['ground_force_l_vy'][grf_start:grf_end]/BW, 
            color='black', lw=lw+1.0)
        ax_grf_z.plot(pgc_grfs,
                      grf_table['ground_force_l_vz'][grf_start:grf_end]/BW,
                      color='black', lw=lw+1.0)

        # experimental coordinates
        coordinates = self.load_table(os.path.join(root_dir, 'resources', 
                'Rajagopal2016', 'coordinates.mot'))
        coords_start = np.argmin(abs(coordinates['time']-self.initial_time))
        coords_end = np.argmin(abs(coordinates['time']-self.final_time))

        time_coords = coordinates['time'][coords_start:coords_end]
        pgc_coords = np.linspace(0, 100, len(time_coords))
        ax_add.plot(pgc_coords,
                    coordinates['hip_adduction_l'][coords_start:coords_end],
                    color='black', lw=lw + 1.0)
        ax_hip.plot(pgc_coords,
                    coordinates['hip_flexion_l'][coords_start:coords_end],
                    color='black', lw=lw + 1.0)
        ax_knee.plot(pgc_coords,
                     coordinates['knee_angle_l'][coords_start:coords_end],
                     color='black', lw=lw + 1.0)
        ax_ankle.plot(pgc_coords,
                      coordinates['ankle_angle_l'][coords_start:coords_end],
                      color='black', lw=lw + 1.0)

        # simulation results
        iterate = zip(
            self.tracking_weights, self.effort_weights,
            self.cmap_indices)

        for i, (tracking_weight, effort_weight, cmap_index) in enumerate(
                iterate):
            color = cmap(cmap_index)
            full_path = self.get_solution_path_fullcycle(root_dir, 
                    tracking_weight, effort_weight)
            full_traj = osim.MocoTrajectory(full_path)

            sol_path = self.get_solution_path(root_dir, tracking_weight,
                                              effort_weight)
            trackingWeight = str(tracking_weight).replace('.', '_')
            effortWeight = str(effort_weight).replace('.', '_')
            sol_table = osim.TimeSeriesTable(sol_path)
            if self.coordinate_tracking:
                trackingCostStr = \
                    sol_table.getTableMetaDataString('objective_state_tracking')
            else:
                trackingCostStr = \
                    sol_table.getTableMetaDataString('objective_marking_tracking')
            trackingCost = float(trackingCostStr) / tracking_weight

            effortCost = 0
            if effort_weight:
                effortCostStr = \
                    sol_table.getTableMetaDataString('objective_control_effort')
                effortCost = float(effortCostStr) / effort_weight
            print(f'effort and tracking costs (track={trackingWeight}, '
                  f'effort={effortWeight}): ', effortCost, trackingCost)

            grf_path = self.get_solution_path_grfs(root_dir,
                    tracking_weight, effort_weight)
            grf_table = self.load_table(grf_path)


            time = full_traj.getTimeMat()
            pgc = np.linspace(0, 100, len(time))

            # pareto front
            ax_time.plot(trackingCost, effortCost, color=color,
                         marker='o')
            ax_time.text(trackingCost, effortCost, self.legend_entries[i])
            # ax_time.bar(i+1, time[-1]-time[0], color=color)
            # ax_time.set_ylim(0, 1.2)
            # ax_time.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
            ax_time.set_xlabel('tracking cost')
            ax_time.set_ylabel('effort cost')
            # ax_time.set_xticks([0, 1, 2, 3])
            # ax_time.set_xticklabels(
            #         ['data', 'track', 'track\n + \neffort', 'effort'])
            ax_time.set_title('COST TRADE-OFF\n', weight='bold', size=title_fs)
            ax_time.set_aspect(1.0/ax_time.get_data_ratio()*0.8, anchor='N')
            utilities.publication_spines(ax_time)

            # ground reaction forces
            ax_grf_x.plot(pgc, grf_table['ground_force_l_vx']/BW, color=color,
                lw=lw)
            ax_grf_x.set_ylabel('horizontal force (BW)')
            ax_grf_x.set_ylim(-0.35, 0.35)
            ax_grf_x.set_yticks([-0.2, 0, 0.2])
            ax_grf_x.set_title('GROUND REACTIONS', weight='bold', 
                    size=title_fs)
            ax_grf_x.set_xticklabels([])

            ax_grf_y.plot(pgc, grf_table['ground_force_l_vy']/BW, color=color,
                lw=lw)
            ax_grf_y.set_ylabel('vertical force (BW)')
            ax_grf_y.set_ylim(0, 1.5)
            ax_grf_y.set_yticks([0, 0.5, 1, 1.5])
            ax_grf_y.set_xticklabels([])

            ax_grf_z.plot(pgc, grf_table['ground_force_l_vz']/BW, color=color,
                          lw=lw)
            ax_grf_z.set_ylabel('transverse force (BW)')
            ax_grf_z.set_xlabel('time (% gait cycle)')

            # kinematics
            rad2deg = 180 / np.pi
            ax_add.plot(pgc, rad2deg*full_traj.getStateMat(
                '/jointset/hip_l/hip_adduction_l/value'), color=color, lw=lw)
            ax_add.set_ylabel('hip adduction (degrees)')
            # ax_add.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
            ax_add.set_xticklabels([])
            ax_add.set_title('KINEMATICS\n', weight='bold', size=title_fs)
            ax_hip.plot(pgc, rad2deg*full_traj.getStateMat(
                    '/jointset/hip_l/hip_flexion_l/value'), color=color, lw=lw)
            ax_hip.set_ylabel('hip flexion (degrees)')
            # ax_hip.set_ylim(-20, 50)
            # ax_hip.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
            ax_hip.set_xticklabels([])
            ax_knee.plot(pgc, rad2deg*full_traj.getStateMat(
                    '/jointset/walker_knee_l/knee_angle_l/value'), color=color,
                    lw=lw)
            ax_knee.set_ylabel('knee flexion (degrees)')
            ax_knee.set_ylim(0, 90)
            ax_knee.set_xticklabels([])
            ax_ankle.plot(pgc, rad2deg*full_traj.getStateMat(
                    '/jointset/ankle_l/ankle_angle_l/value'), color=color, 
                    lw=lw)
            ax_ankle.set_ylabel('ankle dorsiflexion (degrees)')
            ax_ankle.set_xlabel('time (% gait cycle)')
            # ax_ankle.set_ylim(-30, 20)
            # ax_ankle.set_yticks([-0.5, -0.3, -0.1, 0, 0.1, 0.3])


            for ax in ax_list:
                utilities.publication_spines(ax)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])

            # muscle activations
            for im, muscle in enumerate(muscles):
                activation_path = f'/forceset/{muscle[1]}_l/activation'
                ax = muscle[0]
                activation = full_traj.getStateMat(activation_path)
                ax.plot(pgc, activation, color=color, lw=lw)

                # electromyography data
                # TODO: do not assume we want to normalize EMG via simulation 0.
                if i == 0 and 'PSOAS' not in muscle:
                    self.plot(ax, emg['time'],
                              emg[muscle[3]] * np.max(activation), shift=False,
                              fill=True, color='lightgray',
                              label='electromyography')
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1])
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])

                if im < 8:
                    ax.set_xticklabels([])
                ax.text(0.5, 1.2, muscle[2],
                       horizontalalignment='center',
                       verticalalignment='top',
                       transform=ax.transAxes)
                utilities.publication_spines(ax)
                if im == 0:
                    ax.set_title('ACTIVATIONS\n', weight='bold', size=title_fs)
                if im == 8: ax.set_xlabel('time (% gait cycle)')

        fig.align_ylabels([ax_grf_x, ax_grf_y])
        fig.align_ylabels([ax_time, ax_add, ax_hip, ax_knee, ax_ankle])

        # fig.tight_layout()
        fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.065,
                            hspace=200,
                            wspace=0.5)
        fig.savefig(os.path.join(root_dir,
                'figures/motion_tracking_walking.png'), dpi=600)

        with open(os.path.join(root_dir, 'results/'
                'motion_tracking_walking_durations.txt'), 'w') as f:
            iterate = zip(self.tracking_weights, self.effort_weights,
                          self.cmap_indices)
            for tracking_weight, effort_weight, cmap_index in iterate:
                print('cmap_index', cmap_index)
                sol_path = self.get_solution_path(root_dir, tracking_weight,
                        effort_weight)
                trackingWeight = str(tracking_weight).replace('.', '_')
                effortWeight = str(effort_weight).replace('.', '_')
                sol_table = osim.TimeSeriesTable(sol_path)
                duration = sol_table.getTableMetaDataString('solver_duration')
                # Convert duration from seconds to hours.
                duration = float(duration) / 60.0 / 60.0
                print(f'duration (track={trackingWeight}, '
                      f'effort={effortWeight}): ', duration)              
                f.write(f'(track={trackingWeight}, effort={effortWeight}): '
                        f'{duration:.2f}\n')

        iterate = zip(self.tracking_weights, self.effort_weights,
                      self.cmap_indices)
        for tracking_weight, effort_weight, cmap_index in iterate:
            print(f'reserves for track={trackingWeight}, '
                  f'effort={effortWeight}):')
            sol_path = self.get_solution_path(root_dir, tracking_weight,
                                              effort_weight)
            solution = osim.MocoTrajectory(sol_path)
            reserves = self.calc_reserves(root_dir, solution)
            column_labels = reserves.getColumnLabels()
            max_res = -np.inf
            for icol in range(reserves.getNumColumns()):
                column = utilities.toarray(
                    reserves.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                max_res = np.max([max_res, max])
                print(f' max abs {column_labels[icol]}: {max}')
            muscle_mechanics = self.calc_muscle_mechanics(root_dir, solution)
            trackingWeight = str(tracking_weight).replace('.', '_')
            effortWeight = str(effort_weight).replace('.', '_')
            osim.STOFileAdapter.write(muscle_mechanics,
                                      'results/tracking_walking_muscle_mechanics'
                                      f'_trackingWeight{trackingWeight}'
                                      f'_effortWeight{effortWeight}.sto')
        trajectory_filepath = self.get_solution_path_fullcycle(root_dir,
                                                               self.tracking_weights[-1],
                                                               self.effort_weights[-1])

        ref_files = list()
        ref_files.append('tracking_walking_tracked_states.sto')
        iterate = zip(self.tracking_weights[:-1], self.effort_weights[:-1],
                      self.cmap_indices[:-1])
        for tracking_weight, effort_weight, cmap_index in iterate:
            ref_files.append(self.get_solution_path_fullcycle(root_dir,
                                                              tracking_weight,
                                                              effort_weight))

        report = osim.report.Report(model=model, 
                trajectory_filepath=trajectory_filepath, 
                ref_files=ref_files, bilateral=False)
        report.generate()



