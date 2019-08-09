import opensim as osim
import numpy as np
import pylab as pl

pl.ion()


# TODO move this to a Python Example.

def Kirk():
    model = osim.Model()
    body = osim.Body("b", 1, osim.Vec3(0), osim.Inertia(0))
    model.addBody(body);

    joint = osim.SliderJoint("j", model.getGround(), body)
    joint.updCoordinate().setName("coord")
    model.addJoint(joint);

    damper = osim.SpringGeneralizedForce("coord")
    damper.setViscosity(-1.0)
    model.addForce(damper)

    actu = osim.CoordinateActuator("coord")
    model.addForce(actu)
    model.finalizeConnections()

    moco = osim.MocoStudy()
    problem = moco.updProblem()

    problem.setModel(model)
    problem.setTimeBounds(0, 2)
    problem.setStateInfo("/jointset/j/coord/value", [-10, 10], 0, 5)
    problem.setStateInfo("/jointset/j/coord/speed", [-10, 10], 0, 2)
    problem.setControlInfo("/forceset/coordinateactuator", [-50, 50])

    problem.addGoal(osim.MocoControlGoal("effort", 0.5))

    solver = moco.initCasADiSolver();
    solver.set_optim_hessian_approximation("limited-memory");
    solver.set_verbosity(0)
    solver.set_parallel(0)

    def expectedSolution(time):
        A = np.matrix([[-2.0 - 0.5 * np.exp(-2.0) + 0.5 * np.exp(2.0),
                        1.0 - 0.5 * np.exp(-2.0) - 0.5 * np.exp(2.0)],
                       [-1.0 + 0.5 * np.exp(-2.0) + 0.5 * np.exp(2.0),
                        0.5 * np.exp(-2.0) - 0.5 * np.exp(2.0)]])
        b = np.matrix([[5], [2]])
        c = np.linalg.solve(A, b)
        c2 = c[0]
        c3 = c[1]

        def y0(t):
            return (c2 * (-t - 0.5 * np.exp(-t) + 0.5 * np.exp(t)) +
                    c3 * (1.0 - 0.5 * np.exp(-t) - 0.5 * np.exp(t)))

        def y1(t):
            return (c2 * (-1.0 + 0.5 * np.exp(-t) + 0.5 * np.exp(t)) +
                    c3 * (0.5 * np.exp(-t) - 0.5 * np.exp(t)))

        sol = np.empty((len(time), 2))
        for i in range(len(time)):
            sol[i, 0] = y0(time[i])
            sol[i, 1] = y1(time[i])
        return sol

    N = 10
    N_list = []
    error_list = []
    solver_duration = []
    while N < 2000: # 10000:
        solver.set_num_mesh_points(N)
        solution = moco.solve()
        actual_y0 = solution.getStateMat('/jointset/j/coord/value')
        actual_y1 = solution.getStateMat('/jointset/j/coord/speed')
        actual = np.empty((2 * N - 1, 2))
        actual[:, 0] = np.array(actual_y0)
        actual[:, 1] = np.array(actual_y1)
        diff = actual - expectedSolution(solution.getTimeMat())
        error = np.sqrt(np.mean(np.square(diff.flatten())))
        print("N: {}. Error: {}".format(N, error))
        N_list += [N]
        error_list += [error]
        solver_duration += [solution.getSolverDuration()]
        N *= 2

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(N_list, np.array(error_list)) # np.log10(np.array(error_list)))
    ax.set_yscale('log')

    pl.ylabel('root-mean-square error (TODO)')
    fig.add_subplot(2, 1, 2)
    pl.plot(N_list, solver_duration)
    pl.xlabel('number of mesh points')
    pl.ylabel('solver duration (s)')
    pl.show()
    fig.savefig('Kirk.png')

    # TODO add Hermite-Simpson


def brachistochrone():
    model = osim.ModelFactory.createBrachistochrone()

    moco = osim.MocoTool()
    problem = moco.updProblem()

    problem.setModel(model)
    problem.setTimeBounds(0, [0, 10])
    problem.setStateInfo("/brachistochrone/x", [-10, 10], 0, 1)
    problem.setStateInfo("/brachistochrone/y", [-10, 10], 0, 1)
    problem.setStateInfo("/brachistochrone/v", [-10, 10], 0)
    problem.setControlInfo("/brachistochrone", [-100, 100])

    problem.addGoal(osim.MocoFinalTimeGoal())

    # problem.addGoal(osim.MocoControlGoal('effort', 0.01))

    solver = moco.initCasADiSolver();
    solver.set_optim_hessian_approximation("limited-memory");
    solver.set_verbosity(0)
    solver.set_parallel(0)

    # solution = moco.solve()
    # print(solution.getControlsTrajectoryMat())
    # pl.plot(solution.getStateMat('/brachistochrone/x'),
    #         solution.getStateMat('/brachistochrone/y'))
    # pl.figure();
    # pl.plot(solution.getTimeMat(), solution.getControlsTrajectoryMat())
    # pl.show()

    def expectedSolution(time):
        return np.nan

    N = 10
    N_list = []
    error_list = []
    solver_duration = []
    while N < 1000:
        solver.set_num_mesh_points(N)
        solution = moco.solve()
        # actual_y0 = solution.getStateMat('/jointset/j/coord/value')
        # actual_y1 = solution.getStateMat('/jointset/j/coord/speed')
        # actual = np.empty((N, 2))
        # actual[:, 0] = np.array(actual_y0)
        # actual[:, 1] = np.array(actual_y1)
        # diff = actual - expectedSolution(solution.getTimeMat())
        # error = np.sqrt(np.mean(np.square(diff.flatten())))
        error = 1000.0
        duration = solution.getSolverDuration()
        print("N: {}. Error: {}. Duration: {}".format(N, error, duration))
        N_list += [N]
        error_list += [error]
        solver_duration += [duration]
        N *= 2

    pl.figure()
    pl.subplot(2, 1, 1)
    pl.plot(N_list, np.array(error_list)) # np.log10(np.array(error_list)))
    pl.xlabel('number of mesh points')
    pl.ylabel('err')
    pl.subplot(2, 1, 2)
    pl.plot(N_list, solver_duration)
    pl.show()


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


def sit_to_stand():
    def create_tool(model):
        moco = osim.MocoTool()
        solver = moco.initCasADiSolver()
        solver.set_num_mesh_points(25)
        solver.set_dynamics_mode('implicit')
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        solver.set_optim_solver('ipopt')
        solver.set_transcription_scheme('hermite-simpson')
        solver.set_enforce_constraint_derivatives(True)
        solver.set_optim_hessian_approximation('limited-memory')
        solver.set_optim_finite_difference_scheme('forward')

        problem = moco.updProblem()
        problem.setModelCopy(model)
        problem.setTimeBounds(0, 1)
        # TODO remove these for tracking:
        # The position bounds specify that the model should start in a crouch and
        # finish standing up.
        problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value',
                             [-2, 0.5], -2, 0)
        problem.setStateInfo('/jointset/knee_r/knee_angle_r/value',
                             [-2, 0], -2, 0)
        problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value',
                             [-0.5, 0.7], -0.5, 0)
        # The velocity bounds specify that the model coordinates should start and
        # end at zero.
        problem.setStateInfo('/jointset/hip_r/hip_flexion_r/speed',
                             [-50, 50], 0, 0)
        problem.setStateInfo('/jointset/knee_r/knee_angle_r/speed',
                             [-50, 50], 0, 0)
        problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/speed',
                             [-50, 50], 0, 0)

        return moco

    def add_CoordinateActuator(model, coord_name, optimal_force):
        coord_set = model.updCoordinateSet()
        actu = osim.CoordinateActuator()
        actu.setName('tau_' + coord_name)
        actu.setCoordinate(coord_set.get(coord_name))
        actu.setOptimalForce(optimal_force)
        actu.setMinControl(-1)
        actu.setMaxControl(1)
        model.addForce(actu)

    def torque_driven_model():
        model = osim.Model('sitToStand_3dof9musc.osim')
        model.updForceSet().clearAndDestroy()
        model.initSystem()
        add_CoordinateActuator(model, 'hip_flexion_r', 150)
        add_CoordinateActuator(model, 'knee_angle_r', 300)
        add_CoordinateActuator(model, 'ankle_angle_r', 150)
        return model

    def predict():
        moco = create_tool(torque_driven_model())
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        solver = moco.updSolver()
        solver.resetProblem(problem)

        predictSolution = moco.solve()

        predictSolution.write('predictSolution.sto')
        # moco.visualize(predictSolution)

        timeStepSolution = osim.simulateIterateWithTimeStepping(
            predictSolution, problem.getPhase(0).getModel(),
            1e-8)
        timeStepSolution.write('timeStepSolution.sto')
        # moco.visualize(timeStepSolution)

    def track():
        moco = create_tool(torque_driven_model())
        problem = moco.updProblem()
        tracking = osim.MocoStateTrackingGoal()
        tracking.setName('tracking')
        tracking.setReferenceFile('predictSolution.sto')
        tracking.setAllowUnusedReferences(True)
        problem.addGoal(tracking)

        problem.addGoal(osim.MocoControlGoal('effort', 0.01))

        solver = moco.updSolver()
        solver.resetProblem(problem)

        # solver.setGuess(predictSolution)
        trackingSolution = moco.solve()
        trackingSolution.write('trackingSolution.sto')
        # moco.visualize(trackingSolution)

    predict()
    track()

def motion_tracking_walking():

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
    #  - TODO 18 minutes

    # Construct a TableProcessor of the coordinate data and pass it to the
    # tracking tool. TableProcessors can be used in the same way as
    # ModelProcessors by appending TableOperators to modify the base table.
    # A TableProcessor with no operators, as we have here, simply returns the
    # base table.
    track.setStatesReference(osim.TableProcessor("coordinates.sto"))
    track.set_states_global_tracking_weight(10)

    # This setting allows extra data columns contained in the states
    # reference that don't correspond to model coordinates.
    track.set_allow_unused_references(True)

    # Since there is only coordinate position data the states references, this
    # setting is enabled to fill in the missing coordinate speed data using
    # the derivative of splined position data.
    track.set_track_reference_position_derivatives(True)

    # Initial time, final time, and mesh interval.
    track.set_initial_time(0.81)
    track.set_final_time(1.65)
    track.set_mesh_interval(0.08)

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
    solution = moco.solve()
    moco.visualize(solution)

if __name__ == "__main__":
    # Kirk()
    # brachistochrone()
    # suspended_mass()
    # sit_to_stand()
    motion_tracking_walking()
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
