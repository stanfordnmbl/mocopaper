import numpy as np
import re

import opensim as osim

from moco_paper_result import MocoPaperResult

class MotionTrackingWalking(MocoPaperResult):
    self.initial_time = 0.81
    self.half_time = 1.385
    self.final_time = 1.96
    self.inverse_solution_file = \
            'results/motion_prescribed_walking_inverse_solution.sto'
    self.solution_prefix = 'results/motion_tracking_walking_solution'

def create_model_processor(self):

    model = osim.Model(
        'resources/Rajagopal2016/subject_walk_armless_contact_80musc.osim')

    def add_reserve(model, coord, optimal_force, max_control):
            actu = osim.CoordinateActuator(coord)
            if coord.startswith('lumbar'):
                prefix = 'torque_'
            elif coord.startswith('pelvis'):
                prefix = 'residual_'
            else:
                prefix = 'reserve_'
            actu.setName(prefix + coord)
            actu.setOptimalForce(optimal_force)
            actu.setMinControl(-max_control)
            actu.setMaxControl(max_control)
            model.addForce(actu)
    add_reserve(model, 'lumbar_extension', 50, 1)
    add_reserve(model, 'lumbar_bending', 50, 1)
    add_reserve(model, 'lumbar_rotation', 20, 1)
    add_reserve(model, 'pelvis_tilt', 1, 1)
    add_reserve(model, 'pelvis_list', 1, 1)
    add_reserve(model, 'pelvis_rotation', 1, 1)
    add_reserve(model, 'pelvis_tx', 1, 1)
    add_reserve(model, 'pelvis_ty', 1, 1)
    add_reserve(model, 'pelvis_tz', 1, 1)

    modelProcessor = osim.ModelProcessor(model)
    modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
        ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpAddReserves(1, 1, True))

    return modelProcessor

def run_tracking_problem(self, guess, tracking_weight=1.0, control_weight=1.0):

    modelProcessor = self.create_model_processor()

    # Construct the base tracking problem
    # -----------------------------------
    track = osim.MocoTrack()
    track.setModel(modelProcessor);
    track.setStatesReference(
        TableProcessor('resources/Rajagopal2016/coordinates.mot') |
        TabOpLowPassFilter(6) |
        TabOpUseAbsoluteStateNames());
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_states_global_tracking_weight(tracking_weight);
    track.set_control_effort_weight(control_weight);
    track.set_initial_time(0.81);
    track.set_final_time(1.385);
    track.set_mesh_interval(0.02);

    # Customize the base tracking problem
    # -----------------------------------
    study = track.initialize();
    problem = study.updProblem();
    # Ensure that the pelvis starts and ends at the same x-positions from the 
    # measurements, even if the tracking weight is low. Since we are tracking
    # data, we used this instead of an effort-over-distance cost function.
    problem.setStateInfo("/jointset/ground_pelvis/pelvis_tx/value", [],
        0.446, 1.156);

    # Symmetry contraints
    # -------------------
    model = modelProcessor.process()
    model.initSystem();
    symmetry = osim.MocoPeriodicityGoal("symmetry")
    # Symmetric coordinate values (except for pelvis_tx) and speeds.
    for coord in model.getComponentsList():
        if not type(coord) is osim.Coordinate: continue
        if coord.getName().endswith("_r"):
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(0),
                coord.getStateVariableNames().get(0).replace("_r/", "_l/")))
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(1),
                coord.getStateVariableNames().get(1).replace("_r/", "_l/")))
        elif coord.getName().endswith("_l"):
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(0),
                coord.getStateVariableNames().get(0).replace("_l/", "_r/")))
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(1),
                coord.getStateVariableNames().get(1).replace("_l/", "_r/")))
        elif (coord.getName().endswith("_bending") || 
              coord.getName().endswith("_rotation") || 
              coord.getName().endswith("_tz") ||
              coord.getName().endswith("_list")): 
            symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(0)))
            symmetry.addNegatedStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(1)))
        elif not coord.getName().endswith("_tx"):
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(0)))
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                coord.getStateVariableNames().get(1)))
    symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
        "/jointset/groundPelvis/pelvis_tx/speed"))
    # Symmetric muscle activations.
    for muscle in model.getComponentsList():
        if not muscle.getConcreteClassName().endswith('Muscle'): continue
        if muscle.getName().endswith("_r"):
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                muscle.getStateVariableNames().get(0),
                muscle.getStateVariableNames().get(0).replace("_r", "_l")))
        elif muscle.getName().endswith("_l"):
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(
                muscle.getStateVariableNames().get(0),
                muscle.getStateVariableNames().get(0).replace("_l", "_r")))
    problem.addGoal(symmetryGoal)

    # Configure the solver
    # --------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-2)
    solver.setGuess(guess)

    # Solve and print solution.
    # -------------------------
    # Create a full gait cycle trajectory from the periodic solution.
    fullTraj = osim.createPeriodicTrajectory(solution);
    fullTraj.write(f'{self.solution_prefix}_track{tracking_weight}_control{control_weight}.sto')

    # Compute ground reaction forces generated by contact sphere from the full
    # gait cycle trajectory.
    # std::vector<std::string> forceNamesRightFoot{"contactSphereHeel_r",
    #         "contactLateralRearfoot_r", "contactLateralMidfoot_r",
    #         "contactLateralToe_r", "contactMedialToe_r",
    #         "contactMedialMidfoot_r"};
    # std::vector<std::string> forceNamesLeftFoot{"contactSphereHeel_l",
    #         "contactLateralRearfoot_l", "contactLateralMidfoot_l",
    #         "contactLateralToe_l", "contactMedialToe_l",
    #         "contactMedialMidfoot_l"};
    # TimeSeriesTable externalLoads = createExternalLoadsTableForGait(
    #         model, solution, forceNamesRightFoot, forceNamesLeftFoot);

    # writeTableToFile(externalLoads, problemName + "_grfs.sto");

    return solution

def generate_results(self):

    