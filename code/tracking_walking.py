import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import opensim as osim

from moco_paper_result import MocoPaperResult

import utilities
from utilities import plot_joint_moment_breakdown

# TODO: try different foot spacing
# TODO: fix oscillations in GRFz.
# TODO: semimem and gasmed forces are negative.
# TODO: feet are crossing over too much (b/c adductor passive force?)
# TODO: remove reserves from tracking problem?
# TODO: use visualize_trajectory.py to comopare kinematics, programmatically.
#       add skinning through a bunch of markers added to the model? a
#       python algorithm for smoothing the path through these markers?

# TODO: add MocoFrameDistanceConstraint direction.
# TODO: increase value of pelvis_ty in initial guess.
# TODO: after a few days, add net joint moment tracking.

class MocoTrackConfig:
    def __init__(self, name, legend_entry, tracking_weight, effort_weight,
                 cmap_index, flags=[]):
        self.name = name
        self.legend_entry = legend_entry
        self.tracking_weight = tracking_weight
        self.effort_weight = effort_weight
        self.cmap_index = cmap_index
        self.flags = flags


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
        self.cmap = 'nipy_spectral'
        self.configs = [
            MocoTrackConfig(name='torque_driven',
                            legend_entry='torque driven',
                            tracking_weight=10,
                            effort_weight=0.0001,
                            cmap_index=0.95,
                            flags=['torque_driven'])
            # MocoTrackConfig(name='track',
            #                 legend_entry='track',
            #                 tracking_weight=10,
            #                 effort_weight=1,
            #                 cmap_index=0.2),
            # MocoTrackConfig(name='trackeffort',
            #                 legend_entry='track\n+\neffort',
            #                 tracking_weight=0.5,
            #                 effort_weight=1,
            #                 cmap_index=0.5),
            # MocoTrackConfig(name='effort',
            #                 legend_entry='effort',
            #                 tracking_weight=0.01,
            #                 effort_weight=10,
            #                 cmap_index=0.9),
            # MocoTrackConfig(name='trendelenburg',
            #                 legend_entry='trendelenburg',
            #                 tracking_weight=0.5,
            #                 effort_weight=1,
            #                 cmap_index=0.9,
            #                 flags=['trendelenburg']),
            # MocoTrackConfig(name='trendelenburg-lite',
            #                 legend_entry='trendelenburg-lite',
            #                 tracking_weight=0.5,
            #                 effort_weight=1,
            #                 cmap_index=0.5,
            #                 flags=['trendelenburg-lite']),
            # MocoTrackConfig(name='weakvasti',
            #                 legend_entry='weak vasti',
            #                 tracking_weight=0.5,
            #                 effort_weight=1,
            #                 cmap_index=0.2,
            #                 flags=['weakvasti']),
            # MocoTrackConfig(name='weakpfs',
            #                 legend_entry='weak pfs',
            #                 tracking_weight=0.5,
            #                 effort_weight=1,
            #                 cmap_index=0.9,
            #                 flags=['weakpfs']),
            # MocoTrackConfig(name='weaksoleus',
            #                 legend_entry='weak soleus',
            #                 tracking_weight=1,
            #                 effort_weight=10,
            #                 cmap_index=0.6,
            #                 flags=['weaksoleus']),
        ]

    def create_model_processor(self, root_dir, for_inverse=False, config=None):

        flags = []
        if config:
            flags = config.flags

        torque_driven = 'torque_driven' in flags

        model = osim.Model(os.path.join(root_dir,
                'resources/Rajagopal2016/'
                'subject_walk_contact_bounded_80musc.osim'))

        if for_inverse:
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
        else:
            stiffnessMod = 1
            print(f'Modifying contact element stiffnesses by '
                  f'factor {stiffnessMod} and radii...')
            for actu in model.getComponentsList():
                if actu.getConcreteClassName() == 'SmoothSphereHalfSpaceForce':
                    force = osim.SmoothSphereHalfSpaceForce.safeDownCast(actu)
                    force.set_stiffness(stiffnessMod * force.get_stiffness())
                    radius = force.get_contact_sphere_radius()
                    scale = 1
                    if 'Heel' in force.getName():
                        scale = 1.0
                    elif 'Rearfoot' in force.getName():
                        scale = 0.9
                    elif 'Midfoot' in force.getName():
                        scale = 0.7
                    elif 'Toe' in force.getName():
                        scale = 0.5

                    force.set_contact_sphere_radius(scale * radius)
                    print(f'  --> modified contact element {force.getName()}'
                          f' with radius modifier {scale}')

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

        # Upper extremity
        add_reserve(model, 'lumbar_extension', 50, 1)
        add_reserve(model, 'lumbar_bending', 50, 1)
        add_reserve(model, 'lumbar_rotation', 50, 1)
        add_reserve(model, 'arm_flex_r', 15, 1)
        add_reserve(model, 'arm_add_r', 15, 1)
        add_reserve(model, 'arm_rot_r', 15, 1)
        add_reserve(model, 'elbow_flex_r', 15, 1)
        add_reserve(model, 'pro_sup_r', 15, 1)
        add_reserve(model, 'arm_flex_l', 15, 1)
        add_reserve(model, 'arm_add_l', 15, 1)
        add_reserve(model, 'arm_rot_l', 15, 1)
        add_reserve(model, 'elbow_flex_l', 15, 1)
        add_reserve(model, 'pro_sup_l', 15, 1)
        # Lower extremity
        optimal_force = 1
        if for_inverse:
            residuals_max = 250
            add_reserve(model, 'pelvis_tx', optimal_force, residuals_max)
            add_reserve(model, 'pelvis_ty', optimal_force, residuals_max)
            add_reserve(model, 'pelvis_tz', optimal_force, residuals_max)
            add_reserve(model, 'pelvis_tilt', optimal_force, residuals_max)
            add_reserve(model, 'pelvis_list', optimal_force, residuals_max)
            add_reserve(model, 'pelvis_rotation', optimal_force, residuals_max)
        reserves_max = 250 if for_inverse or torque_driven else 1
        add_reserve(model, 'hip_flexion_r', optimal_force, reserves_max)
        add_reserve(model, 'knee_angle_r', optimal_force, reserves_max)
        add_reserve(model, 'ankle_angle_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_flexion_l', optimal_force, reserves_max)
        add_reserve(model, 'knee_angle_l', optimal_force, reserves_max)
        add_reserve(model, 'ankle_angle_l', optimal_force, reserves_max)
        add_reserve(model, 'hip_adduction_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_adduction_l', optimal_force, reserves_max)
        add_reserve(model, 'hip_rotation_r', optimal_force, reserves_max)
        add_reserve(model, 'hip_rotation_l', optimal_force, reserves_max)

        def add_device(model, coord):
            actu = osim.ActivationCoordinateActuator()
            actu.set_coordinate(coord)
            actu.setName(f'device_{coord}')
            actu.setOptimalForce(1000.0)
            actu.setMinControl(-1.0)
            actu.setMaxControl(1.0)
            model.addForce(actu)
        if 'assistankle' in flags:
            add_device(model, 'ankle_angle_r')
            add_device(model, 'ankle_angle_l')

        if 'weaksoleus' in flags:
            soleus_r = model.updMuscles().get('soleus_r')
            soleus_r.set_max_isometric_force(
                0.25 * soleus_r.get_max_isometric_force())
            soleus_l = model.updMuscles().get('soleus_l')
            soleus_l.set_max_isometric_force(
                0.25 * soleus_l.get_max_isometric_force())
        if 'weakvasti' in flags:
            for muscle in ['vasmed', 'vasint', 'vaslat']:
                for side in ['_l', '_r']:
                    musc = model.updMuscles().get('%s%s' % (muscle, side))
                    musc.set_max_isometric_force(
                        0.10 * musc.get_max_isometric_force())
        if 'trendelenburg-lite' in flags:
            for muscle in ['glmin1','glmin2','glmin3','glmed1','glmed2','glmed3',
                    'glmax1','glmax2','glmax3','addbrev', 'addlong', 'addmagDist', 
                    'addmagIsch', 'addmagMid', 'addmagProx', 'piri', 'bflh', 
                    'sart', 'grac']:
                for side in ['_l', '_r']:
                    musc = model.updMuscles().get('%s%s' % (muscle, side))
                    musc.set_max_isometric_force(
                        0.10 * musc.get_max_isometric_force())
        if 'trendelenburg' in flags:
            for muscle in ['glmin1','glmin2','glmin3','glmed1','glmed2','glmed3',
                    'glmax1','glmax2','glmax3','addbrev', 'addlong', 'addmagDist', 
                    'addmagIsch', 'addmagMid', 'addmagProx', 'iliacus', 'psoas',
                    'piri', 'grac', 'bflh', 'recfem', 'sart', 'semimem', 
                    'semiten', 'tfl']:
                for side in ['_l', '_r']:
                    musc = model.updMuscles().get('%s%s' % (muscle, side))
                    musc.set_max_isometric_force(
                        0.10 * musc.get_max_isometric_force())
        if 'weakpfs' in flags:
            for muscle in ['soleus', 'gasmed', 'gaslat', 'perbrev', 'perlong',
                           'tibpost', 'fdl', 'fhl']:
                for side in ['_l', '_r']:
                    musc = model.updMuscles().get('%s%s' % (muscle, side))
                    musc.set_max_isometric_force(
                        0.10 * musc.get_max_isometric_force())

        modelProcessor = osim.ModelProcessor(model)

        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l', 'radius_hand_r',
             'radius_hand_l']))

        if torque_driven:
            modelProcessor.append(osim.ModOpRemoveMuscles())
        else:
            modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
            modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        # modelProcessor.append(osim.ModOpPassiveFiberStrainAtOneNormForceDGF(1.0))
        # modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        # modelProcessor.append(osim.ModOpFiberDampingDGF(0.01))
        if for_inverse:
            ext_loads_xml = os.path.join(root_dir,
                    'resources/Rajagopal2016/grf_walk.xml')
            modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))

        if torque_driven:
            return modelProcessor
        else:
            model = modelProcessor.process()
            model.initSystem()
            muscles = model.updMuscles()
            for imusc in np.arange(muscles.getSize()):
                muscle = muscles.get(int(imusc))
                muscName = muscle.getName()
                if ('gas' in muscName) or ('soleus' in muscName):
                    muscle.set_ignore_tendon_compliance(False)

                muscleDGF = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
                if 'soleus' in muscName:
                    muscleDGF.set_ignore_passive_fiber_force(True)
                    muscleDGF.set_passive_fiber_strain_at_one_norm_force(10.0)
                elif 'gas' in muscName:
                    muscleDGF.set_passive_fiber_strain_at_one_norm_force(0.4)
                    muscleDGF.set_fiber_damping(0.001)
                elif (('vas' in muscName) or
                      ('recfem' in muscName) or
                      ('semi' in muscName)):
                    muscleDGF.set_passive_fiber_strain_at_one_norm_force(7.0)
                elif (('tib' in muscName) or
                      ('ehl' in muscName) or
                      ('edl' in muscName)) :
                    muscleDGF.set_passive_fiber_strain_at_one_norm_force(7.0)

            modelProcessorTendonCompliance = osim.ModelProcessor(model)
            modelProcessorTendonCompliance.append(
                    osim.ModOpUseImplicitTendonComplianceDynamicsDGF())

            return modelProcessorTendonCompliance

    def get_solution_path(self, root_dir, config):
        return os.path.join(root_dir,
                            f'{self.tracking_solution_relpath_prefix}'
                            + config.name + '.sto')

    def get_solution_path_fullcycle(self, root_dir, config):
        return os.path.join(root_dir,
                            f'{self.tracking_solution_relpath_prefix}'
                            + config.name + '_fullcycle.sto')

    def get_solution_path_grfs(self, root_dir, config):
        return os.path.join(root_dir,
                            f'{self.tracking_solution_relpath_prefix}'
                            + config.name + '_fullcycle_grfs.sto')

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

    def calc_reserves(self, root_dir, config, solution):
        modelProcessor = self.create_model_processor(root_dir, config=config)
        model = modelProcessor.process()
        output = osim.analyze(model, solution, ['.*reserve.*actuation'])
        return output

    def calc_muscle_mechanics(self, root_dir, config, solution):
        modelProcessor = self.create_model_processor(root_dir, config=config)
        model = modelProcessor.process()
        outputList = list()

        for output in ['normalized_fiber_length', 'normalized_fiber_velocity', 
                       'passive_fiber_force']:
            for imusc in range(model.getMuscles().getSize()):
                musc = model.updMuscles().get(imusc)
                outputList.append(f'.*{musc.getName()}.*\|{output}')

        outputs = osim.analyze(model, solution, outputList)

        return outputs

    def run_inverse_problem(self, root_dir):

        modelProcessor = self.create_model_processor(root_dir,
                                                     for_inverse=True,
                                                     config=self.configs[0])
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

    def run_tracking_problem(self, root_dir, config):

        flags = []
        if config:
            flags = config.flags

        torque_driven = 'torque_driven' in flags

        modelProcessor = self.create_model_processor(root_dir,
                                                     for_inverse=False,
                                                     config=config)
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
                config.tracking_weight / (2 * model.getNumCoordinates()))
            # Don't track some pelvis coordinates to avoid poor walking motion
            # solutions.
            stateWeights = osim.MocoWeightSet()
            weightList = list()
            weightList.append(('/jointset/ground_pelvis/pelvis_ty/value', 0))
            weightList.append(('/jointset/ground_pelvis/pelvis_tz/value', 0))
            weightList.append(('/jointset/ground_pelvis/pelvis_list/value', 0.1))
            weightList.append(('/jointset/ground_pelvis/pelvis_tilt/value', 0.1))
            weightList.append(('/jointset/ground_pelvis/pelvis_rotation/value', 0.1))
            weightList.append(('/jointset/back/lumbar_rotation/value', 0.1))
            if not torque_driven:
                weightList.append(('/jointset/hip_r/hip_rotation_r/value', 0))
                weightList.append(('/jointset/hip_r/hip_adduction_r/value', 0))
                weightList.append(('/jointset/hip_l/hip_rotation_l/value', 0))
                weightList.append(('/jointset/hip_l/hip_adduction_l/value', 0))
            for weight in weightList:
                stateWeights.cloneAndAppend(osim.MocoWeight(weight[0], weight[1]))
            track.set_states_weight_set(stateWeights)
            track.set_apply_tracked_states_to_guess(True)
        else:
            track.setMarkersReferenceFromTRC(os.path.join(root_dir,
                    'resources/Rajagopal2016/markers.trc'))
            track.set_markers_global_tracking_weight(
                config.tracking_weight / (2 * model.getNumMarkers()))
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
        track.set_control_effort_weight(config.effort_weight / numForces)
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
        # problem.setStateInfo('/jointset/back/lumbar_rotation/value', [], 0.04)
        # problem.setStateInfo('/jointset/ground_pelvis/pelvis_tx/value', [], 0.446)
        problem.setStateInfo('/jointset/ground_pelvis/pelvis_tilt/value', [], -0.01)
        # problem.setStateInfo('/jointset/ground_pelvis/pelvis_list/value', [], 0)
        # problem.setStateInfo('/jointset/ground_pelvis/pelvis_rotation/value', [], 0)

        # Update the control effort goal to a cost of transport type cost.
        effort = osim.MocoControlGoal().safeDownCast(
                problem.updGoal('control_effort'))
        effort.setDivideByDisplacement(True)
        # Weight residual and reserve actuators low in the effort cost since
        # they are already weak.
        if config.effort_weight:
            for actu in model.getComponentsList():
                actuName = actu.getName()
                if actu.getConcreteClassName().endswith('Actuator'):
                    effort.setWeightForControl(actu.getAbsolutePathString(),
                        0.001)

            if not torque_driven:
                for muscle in ['psoas', 'iliacus']:
                    for side in ['l', 'r']:
                        effort.setWeightForControl(
                            '/forceset/%s_%s' % (muscle, side), 0.5)

        speedGoal = osim.MocoAverageSpeedGoal('speed')
        speedGoal.set_desired_average_speed(1.235)
        problem.addGoal(speedGoal)

        # MocoFrameDistanceConstraint
        # ---------------------------
        if self.coordinate_tracking:
            distanceConstraint = osim.MocoFrameDistanceConstraint()
            distanceConstraint.setName('distance_constraint')
            # Step width is 0.13 * leg_length
            # distance = 0.10 # TODO Should be closer to 0.11.
            # Donelan JM, Kram R, Kuo AD. Mechanical and metabolic determinants
            # of the preferred step width in human walking.
            # Proc Biol Sci. 2001;268(1480):1985–1992.
            # doi:10.1098/rspb.2001.1761
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1088839/
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/calcn_l', '/bodyset/calcn_r', 0.09, np.inf))
            distanceConstraint.addFramePair(
                    osim.MocoFrameDistanceConstraintPair(
                    '/bodyset/toes_l', '/bodyset/toes_r', 0.06, np.inf))
            # distanceConstraint.addFramePair(
            #         osim.MocoFrameDistanceConstraintPair(
            #         '/bodyset/calcn_l', '/bodyset/toes_r', distance, np.inf))
            # distanceConstraint.addFramePair(
            #         osim.MocoFrameDistanceConstraintPair(
            #         '/bodyset/toes_l', '/bodyset/calcn_r', distance, np.inf))
            distanceConstraint.setProjection("vector")
            distanceConstraint.setProjectionVector(osim.Vec3(0, 0, 1))
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
                  coordName.endswith('_tz') or
                  coordName.endswith('_list')):
                # This does not include hip rotation,
                # because that ends with _l or _r.
                symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                    coordValue))
                symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                    coordSpeed))
            elif coordName.endswith('_rotation'):
                symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
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
            contactTracking.setExternalLoadsFile(os.path.join(root_dir,
                'resources/Rajagopal2016/grf_walk.xml'))
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
        # Create a guess compatible with this problem.
        guess = solver.createGuess()
        # Load the inverse problem solution and set its states and controls
        # trajectories to the guess.
        inverseSolution = osim.MocoTrajectory(
            os.path.join(root_dir, self.inverse_solution_relpath))
        inverseStatesTable = inverseSolution.exportToStatesTable()
        for stateLabel in inverseStatesTable.getColumnLabels():
            if ('pelvis' in stateLabel) and stateLabel.endswith('/activation'):
                inverseStatesTable.removeColumn(stateLabel)
        guess.insertStatesTrajectory(inverseStatesTable, True)
        # Changing this initial guess has a large negative effect on the
        # solution! Lots of knee flexion.
        # guess.setState('/jointset/ground_pelvis/pelvis_ty/value',
        #                osim.Vector(guess.getNumTimes(), 1.01))
        inverseControlsTable = inverseSolution.exportToControlsTable()
        for controlLabel in inverseControlsTable.getColumnLabels():
            if 'pelvis' in controlLabel:
                inverseControlsTable.removeColumn(controlLabel)
        guess.insertControlsTrajectory(inverseControlsTable, True)
        solver.setGuess(guess)

        # Solve and print solution.
        # -------------------------
        solution = study.solve()
        solution.write(self.get_solution_path(root_dir, config))

        # solution = osim.MocoTrajectory(self.get_solution_path(root_dir, config))

        # Create a full gait cycle trajectory from the periodic solution.
        addPatterns = [".*pelvis_tx/value"]
        negatePatterns = [".*pelvis_list(?!/value).*",
                          ".*pelvis_rotation.*",
                          # ".*pelvis_tilt(?!/value).*",
                          ".*pelvis_tz(?!/value).*",
                          ".*lumbar_bending(?!/value).*",
                          ".*lumbar_rotation.*"]
        negateAndShiftPatterns = [".*pelvis_list/value",
                                  ".*pelvis_tz/value",
                                  # ".*pelvis_tilt/value",
                                  ".*lumbar_bending/value"]
        fullTraj = osim.createPeriodicTrajectory(solution, addPatterns,
            negatePatterns, negateAndShiftPatterns)
        fullTraj.write(self.get_solution_path_fullcycle(root_dir, config))

        # Compute ground reaction forces generated by contact sphere from the 
        # full gait cycle trajectory.
        externalLoads = osim.createExternalLoadsTableForGait(
                model, fullTraj, forceNamesRightFoot, forceNamesLeftFoot)
        osim.writeTableToFile(externalLoads,
                              self.get_solution_path_grfs(root_dir, config))

    def parse_args(self, args):
        self.skip_inverse = False
        self.coordinate_tracking = False
        self.contact_tracking = False
        self.visualize = False
        if len(args) == 0: return
        print('Received arguments {}'.format(args))
        if 'skip-inverse' in args:
            self.skip_inverse = True
        if 'coordinate-tracking' in args:
            self.coordinate_tracking = True
        if 'contact-tracking' in args:
            self.contact_tracking = True
        if 'visualize' in args:
            self.visualize = True

    def generate_results(self, root_dir, args):
        self.parse_args(args)

        # Run inverse problem to generate first initial guess.
        if not self.skip_inverse:
            self.run_inverse_problem(root_dir)

        # Run tracking problem.
        for config in self.configs:
            self.run_tracking_problem(root_dir, config)

    def report_results(self, root_dir, args):
        self.parse_args(args)

        modelProcessor = self.create_model_processor(root_dir, 
            config=self.configs[0])
        model = modelProcessor.process()
        state = model.initSystem()
        mass = model.getTotalMass(state)
        gravity = model.getGravity()
        BW = mass*abs(gravity[1])

        if self.visualize:
            for config in self.configs:
                solution = osim.MocoTrajectory(
                    self.get_solution_path_fullcycle(root_dir, config))
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
        for i, config in enumerate(self.configs):
            color = cmap(config.cmap_index)
            full_path = self.get_solution_path_fullcycle(root_dir, config)
            full_traj = osim.MocoTrajectory(full_path)

            sol_path = self.get_solution_path(root_dir, config)
            sol_table = osim.TimeSeriesTable(sol_path)
            if self.coordinate_tracking:
                trackingCostStr = \
                    sol_table.getTableMetaDataString('objective_state_tracking')
            else:
                trackingCostStr = \
                    sol_table.getTableMetaDataString(
                        'objective_marker_tracking')
            trackingCost = float(trackingCostStr) / config.tracking_weight

            effortCost = 0
            if config.effort_weight:
                effortCostStr = \
                    sol_table.getTableMetaDataString('objective_control_effort')
                effortCost = float(effortCostStr) / config.effort_weight
            print(f'effort and tracking costs (config: {config.name}): ',
                  effortCost, trackingCost)

            grf_path = self.get_solution_path_grfs(root_dir, config)
            grf_table = self.load_table(grf_path)


            time = full_traj.getTimeMat()
            pgc = np.linspace(0, 100, len(time))

            # pareto front
            ax_time.plot(trackingCost, effortCost, color=color,
                         marker='o')
            ax_time.text(trackingCost, effortCost, config.legend_entry)
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
            ax_grf_z.set_ylim(-0.35, 0.35)
            ax_grf_z.set_yticks([-0.2, 0, 0.2])

            # kinematics
            rad2deg = 180 / np.pi
            ax_add.plot(pgc, rad2deg*full_traj.getStateMat(
                '/jointset/hip_l/hip_adduction_l/value'), color=color, lw=lw)
            ax_add.set_ylabel('hip adduction (degrees)')
            ax_add.set_xticklabels([])
            ax_add.set_title('KINEMATICS\n', weight='bold', size=title_fs)
            ax_hip.plot(pgc, rad2deg*full_traj.getStateMat(
                    '/jointset/hip_l/hip_flexion_l/value'), color=color, lw=lw)
            ax_hip.set_ylabel('hip flexion (degrees)')
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


            for ax in ax_list:
                utilities.publication_spines(ax)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])

            # muscle activations
            if not 'torque_driven' in config.flags:
                for im, muscle in enumerate(muscles):
                    activation_path = f'/forceset/{muscle[1]}_l/activation'
                    ax = muscle[0]
                    activation = full_traj.getStateMat(activation_path)
                    ax.plot(pgc, activation, color=color, lw=lw)

                    # electromyography data
                    # TODO: do not assume we want to normalize EMG via 
                    # simulation 0.
                    if i == 0 and 'PSOAS' not in muscle:
                        self.plot(ax, emg['time'],
                                  emg[muscle[3]] * np.max(activation), 
                                  shift=False, fill=True, color='lightgray',
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
                        ax.set_title('ACTIVATIONS\n', weight='bold', 
                                     size=title_fs)
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
            for config in self.configs:
                sol_path = self.get_solution_path(root_dir, config)
                sol_table = osim.TimeSeriesTable(sol_path)
                duration = sol_table.getTableMetaDataString('solver_duration')
                # Convert duration from seconds to hours.
                duration = float(duration) / 60.0 / 60.0
                print(f'duration (config {config.name}): ', duration)
                f.write(f'(config {config.name}): {duration:.2f}\n')

        for config in self.configs:
            print(f'reserves for config {config.name}:')
            sol_path = self.get_solution_path_fullcycle(root_dir, config)
            solution = osim.MocoTrajectory(sol_path)
            reserves = self.calc_reserves(root_dir, config, solution)
            column_labels = reserves.getColumnLabels()
            max_res = -np.inf
            for icol in range(reserves.getNumColumns()):
                column = utilities.toarray(
                    reserves.getDependentColumnAtIndex(icol))
                max = np.max(np.abs(column))
                max_res = np.max([max_res, max])
                print(f' max abs {column_labels[icol]}: {max}')
            muscle_mechanics = self.calc_muscle_mechanics(root_dir, config,
                                                          solution)

            fpath = os.path.join(root_dir, 
                    'results/tracking_walking_muscle_mechanics_'
                   f'{config.name}.sto')
            osim.STOFileAdapter.write(muscle_mechanics, fpath)

            # Generate joint moment breakdown.
            modelProcessor = self.create_model_processor(root_dir,
                                                         config=config)
            model = modelProcessor.process()
            print(f'Generating joint moment breakdown for {config.name}.')
            coords = [
                '/jointset/hip_l/hip_flexion_l',
                '/jointset/walker_knee_l/knee_angle_l',
                '/jointset/ankle_l/ankle_angle_l'
            ]
            fig = plot_joint_moment_breakdown(model, solution,
                                              coords)
            fpath = os.path.join(root_dir,
                                 'results/motion_tracking_walking_' +
                                 f'breakdown_{config.name}.png')
            fig.savefig(fpath, dpi=600)

        # Generate PDF report.
        # --------------------
        trajectory_filepath = self.get_solution_path_fullcycle(root_dir,
                                                               self.configs[-1]
                                                               )
        ref_files = list()
        ref_files.append('tracking_walking_tracked_states.sto')
        report_suffix = ''
        for config in self.configs[:-1]:
            ref_files.append(self.get_solution_path_fullcycle(root_dir, config))
            report_suffix += '_' + config.name
        report_suffix += '_' + self.configs[-1].name

        report_output = f'motion_tracking_walking{report_suffix}_report.pdf'
        report = osim.report.Report(model=model,
                                    trajectory_filepath=trajectory_filepath,
                                    ref_files=ref_files, bilateral=False,
                                    output=report_output)
        report.generate()
