from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import opensim as osim

class MocoPaperResult(ABC):
    @abstractmethod
    def generate_results(self):
        pass
    @abstractmethod
    def report_results(self):
        pass

def suspended_mass():
    width = 0.2

    def buildModel():
        model = osim.ModelFactory.createPlanarPointMass()
        body = model.getBodySet().get("body")
        model.updForceSet().clearAndDestroy()
        model.finalizeFromProperties()

        actuL = osim.DeGrooteFregly2016Muscle()
        actuL.setName("left")
        actuL.set_max_isometric_force(20)
        actuL.set_optimal_fiber_length(.20)
        actuL.set_tendon_slack_length(0.10)
        actuL.set_pennation_angle_at_optimal(0.0)
        actuL.set_ignore_tendon_compliance(True)
        actuL.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(-width, 0, 0))
        actuL.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuL)

        actuM = osim.DeGrooteFregly2016Muscle()
        actuM.setName("middle")
        actuM.set_max_isometric_force(20)
        actuM.set_optimal_fiber_length(0.09)
        actuM.set_tendon_slack_length(0.1)
        actuM.set_pennation_angle_at_optimal(0.0)
        actuM.set_ignore_tendon_compliance(True)
        actuM.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(0, 0, 0))
        actuM.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuM)

        actuR = osim.DeGrooteFregly2016Muscle()
        actuR.setName("right");
        actuR.set_max_isometric_force(40)
        actuR.set_optimal_fiber_length(.21)
        actuR.set_tendon_slack_length(0.09)
        actuR.set_pennation_angle_at_optimal(0.0)
        actuR.set_ignore_tendon_compliance(True)
        actuR.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(+width, 0, 0))
        actuR.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuR)

        model.finalizeConnections();
        return model

    def predict():
        moco = osim.MocoTool()
        problem = moco.updProblem()
        problem.setModel(buildModel())
        problem.setTimeBounds(0, 0.5)
        problem.setStateInfo("/jointset/tx/tx/value", [-0.03, 0.03], -0.03,
                             0.03)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * width, 0], -width,
                             -width + 0.05)
        problem.setStateInfo("/jointset/tx/tx/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/jointset/ty/ty/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.setControlInfo("/forceset/left", [0, 1])
        problem.setControlInfo("/forceset/middle", [0, 1])
        problem.setControlInfo("/forceset/right", [0, 1])

        problem.addGoal(osim.MocoControlGoal())

        solver = moco.initCasADiSolver()
        solver.set_num_mesh_points(25)
        # solver.set_transcription_scheme("hermite-simpson")
        solution = moco.solve()

        # moco.visualize(solution)

        # pl.figure()
        # pl.plot(solution.getTimeMat(), solution.getStatesTrajectoryMat())
        # pl.legend(solution.getStateNames())
        # pl.figure()
        # pl.plot(solution.getTimeMat(), solution.getControlsTrajectoryMat())
        # pl.legend(solution.getControlNames())
        # pl.show()
        return solution

    def track(prediction):
        moco = osim.MocoTool()
        problem = moco.updProblem()
        problem.setModel(buildModel())
        problem.setTimeBounds(0, 0.5)
        problem.setStateInfo("/jointset/tx/tx/value", [-0.03, 0.03], -0.03,
                             0.03)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * width, 0], -width,
                             -width + 0.05)
        problem.setStateInfo("/jointset/tx/tx/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/jointset/ty/ty/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.setControlInfo("/forceset/left", [0, 1])
        problem.setControlInfo("/forceset/middle", [0, 1])
        problem.setControlInfo("/forceset/right", [0, 1])

        tracking = osim.MocoStateTrackingGoal("tracking")
        tracking.setReference(prediction.exportToStatesTable())
        problem.addGoal(tracking)
        effort = osim.MocoControlGoal("effort")
        effort.setExponent(4)
        problem.addGoal(effort)

        solver = moco.initCasADiSolver()
        solver.set_num_mesh_points(25)
        # solver.set_transcription_scheme("hermite-simpson")
        solution = moco.solve()

        # moco.visualize(solution)
        #
        # pl.figure()
        # pl.plot(solution.getTimeMat(), solution.getStatesTrajectoryMat())
        # pl.legend(solution.getStateNames())
        # pl.figure()
        # pl.plot(solution.getTimeMat(), solution.getControlsTrajectoryMat())
        # pl.legend(solution.getControlNames())
        # pl.show()
        return solution

    predictSolution = predict()
    trackSolution = track(predictSolution)

    pl.figure()
    pl.plot(predictSolution.getTimeMat(),
            predictSolution.getControlsTrajectoryMat())
    pl.plot(trackSolution.getTimeMat(),
            trackSolution.getControlsTrajectoryMat())
    pl.legend(predictSolution.getControlNames())
    pl.show()

    # TODO surround the point with muscles and maximize distance traveled.


class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        self.initial_time = 0.81
        self.final_time = 1.65
        self.mesh_interval = 0.10
        self.mocotrack_solution_file = 'motion_tracking_walking_track_solution.sto'
        self.mocoinverse_solution_file = 'motion_tracking_walking_inverse_solution.sto'
    def generate_results(self):
        # Create and name an instance of the MocoTrack tool.
        track = osim.MocoTrack()
        track.setName("muscle_driven_state_tracking")

        # Construct a ModelProcessor and set it on the tool. The default
        # muscles in the model are replaced with optimization-friendly
        # DeGrooteFregly2016Muscles, and adjustments are made to the default muscle
        # parameters.
        modelProcessor = osim.ModelProcessor("subject_walk_armless.osim")
        modelProcessor.append(osim.ModOpAddExternalLoads("grf_walk.xml"))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        # Only valid for DeGrooteFregly2016Muscles.
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        # Only valid for DeGrooteFregly2016Muscles.
        modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
        track.setModel(modelProcessor)

        # TODO:
        #  - more mesh points,
        #  - avoid removing muscle passive forces
        #  - move example to Python Examples rather than Article folder.
        #  - play with weights between tracking and effort.
        #  - add plot of muscle activity for 9 muscles.
        #  - look for the script that plots muscle activity for the
        #    sit-to-stand example
        #  - simulate an entire gait cycle.
        #  - for the preprint, use gait10dof18musc.
        #  - simulate many gait cycles? show that the code works fine on a suite
        #    of subjects (5 subjects?). **create a separate git repository.
        #  - report duration to solve the problem.
        #  - figure could contain still frames of the model throughout the motion.

        # Construct a TableProcessor of the coordinate data and pass it to the
        # tracking tool. TableProcessors can be used in the same way as
        # ModelProcessors by appending TableOperators to modify the base table.
        # A TableProcessor with no operators, as we have here, simply returns the
        # base table.
        coordinates = osim.TableProcessor("coordinates.sto")
        track.setStatesReference(coordinates)
        track.set_states_global_tracking_weight(10)

        # This setting allows extra data columns contained in the states
        # reference that don't correspond to model coordinates.
        track.set_allow_unused_references(True)

        # Since there is only coordinate position data the states references, this
        # setting is enabled to fill in the missing coordinate speed data using
        # the derivative of splined position data.
        track.set_track_reference_position_derivatives(True)

        # Initial time, final time, and mesh interval.
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(self.mesh_interval)

        # Instead of calling solve(), call initialize() to receive a pre-configured
        # MocoStudy object based on the settings above. Use this to customize the
        # problem beyond the MocoTrack interface.
        moco = track.initialize()

        # Get a reference to the MocoControlCost that is added to every MocoTrack
        # problem by default.
        problem = moco.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

        # Put a large weight on the pelvis CoordinateActuators, which act as the
        # residual, or 'hand-of-god', forces which we would like to keep as small
        # as possible.
        model = modelProcessor.process()
        model.initSystem()
        forceSet = model.getForceSet()
        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            if 'pelvis' in str(forcePath):
                effort.setWeightForControl(forcePath, 10)

        # Solve and visualize.
        moco.printToXML('motion_tracking_walking.omoco')
        solution = moco.solve()
        solution.print(self.mocotrack_solution_file)
        # moco.visualize(solution)

        # TODO plotting should happen separately from generating the results.
        cmc = osim.CMCTool()
        cmc.setName('motion_tracking_walking_cmc')
        cmc.setModel(model)
        # TODO filter:
        cmc.setDesiredKinematicsFileName('coordinates.sto')
        # cmc.setLowpassCutoffFrequency(6)
        cmc.run()

        # TODO compare to MocoInverse.
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(self.mesh_interval)
        solution = inverse.solve()
        solution.getMocoSolution().print(self.mocoinverse_solution_file)


    def report_results(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        solution = osim.MocoTrajectory(self.mocotrack_solution_file)
        ax.plot(solution.getTimeMat(), solution.getStatesTrajectoryMat())

        fig.savefig('motion_tracking_walking.eps')
        fig.savefig('motion_tracking_walking.png', dpi=600)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate results for the"
                                                 "OpenSim Moco publication.")
    parser.add_argument('--no-generate', dest='generate', action='store_false',
                        help='Skip generating the results; only report.')
    parser.set_defaults(generate=True)
    args = parser.parse_args()

    results = [MotionTrackingWalking()]
    for result in results:
        if args.generate:
            result.generate_results()
        result.report_results()

    # motion_prediction_walking()
    # assisted_walking()
# TODO linear tangent steering has analytical solution Example 4.1 Betts, and Example 4.5


# 2 dof 3 muscles, predict, time-stepping, and track. add noise!!!

"""
Results
We show:
Verification
Types of problems:
Prediction
Tracking
Muscle redundancy
Parameter optimization
Modeling features:
Kinematic constraints
Minimizing joint loading
For Aim 1: Verification
Solve a problem with a known analytical solution.
Effect of mesh size on accuracy of solution and duration to solve.
Trapezoidal
Hermite-Simpson
Kinematic constraints
Implementing the method to solve this problem requires careful consideration, and so we present verification so that users are confident we handle this type of problem correctly.
Double pendulum with point on line:
Start from intermediate pose and minimize a certain component of the reaction: ensure the model goes to the correct pose.
Compute what the joint reaction should be (analytically) and check that that’s what it is in the simulation (the cost function).
For Aim 1 (verification) and Aim 2 (science)
Lifting from crouch
Describe model
The model contains a kinematic constraint to move the patella, improving the accuracy of the moment arms for quadriceps muscles. 
Predict
Forward simulation gives same results as direct collocation.
Track
Track the predicted motion to recover muscle activity.
Muscle redundancy (using MocoInverse)
Compare inverse solution to CMC activations.
Compare runtime to CMC.
How much faster than tracking?
Add an assistive device for sit-to-stand (predictive simulation).
Optimize a device parameter.
Do kinematics change?
Shoulder flexion prediction
TODO
We have shown verification, as is necessary for any software, and validation for specific examples. However, users may still obtain invalid results on their own problems. It is essential that researchers always perform their own validation.
"""
