
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
        model = osim.Model('squat_to_stand_4dof9musc.osim')
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
