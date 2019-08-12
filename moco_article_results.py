from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        self.initial_time = 0.73
        self.final_time = 1.795
        self.mocotrack_solution_file = 'motion_tracking_walking_track_solution.sto'
        self.mocoinverse_solution_file = 'motion_tracking_walking_inverse_solution.sto'
        self.side = 'r'

    def generate_results(self):
        # Create and name an instance of the MocoTrack tool.
        track = osim.MocoTrack()
        track.setName("motion_tracking_walking")

        # Construct a ModelProcessor and set it on the tool. The default
        # muscles in the model are replaced with optimization-friendly
        # DeGrooteFregly2016Muscles, and adjustments are made to the default muscle
        # parameters.
        modelProcessor = osim.ModelProcessor(
            "resources/ArnoldSubject02Walk3/subject02_armless.osim")
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
        modelProcessor.append(osim.ModOpAddReserves(10))
        modelProcessor.process().printToXML("subject02_armless_for_cmc.osim")
        ext_loads_xml = "resources/ArnoldSubject02Walk3/external_loads.xml"
        modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))
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
        #  - make sure residuals and reserves are small.

        coordinates = osim.TableProcessor(
            "resources/ArnoldSubject02Walk3/subject02_walk3_ik_solution.mot")
        coordinates.append(osim.TabOpLowPassFilter(6))
        track.setStatesReference(coordinates)
        track.set_states_global_tracking_weight(10)

        # This setting allows extra data columns contained in the states
        # reference that don't correspond to model coordinates.
        track.set_allow_unused_references(True)

        track.set_track_reference_position_derivatives(True)

        # Initial time, final time, and mesh interval.
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(0.01)

        moco = track.initialize()

        problem = moco.updProblem()
        effort = osim.MocoControlGoal.safeDownCast(
            problem.updGoal("control_effort"))

        model = modelProcessor.process()
        model.initSystem()
        forceSet = model.getForceSet()
        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            if 'pelvis' in str(forcePath):
                effort.setWeightForControl(forcePath, 10)

        # Solve and visualize.
        moco.printToXML('motion_tracking_walking.omoco')
        # solution = moco.solve()
        # solution.write(self.mocotrack_solution_file)
        # moco.visualize(solution)

        # tasks = osim.CMC_TaskSet()
        # for coord in model.getCoordinateSet():
        #     task = osim.CMC_Joint()
        #     task.setName(coord.getName())
        #     task.setCoordinateName(coord.getName())
        #     task.setKP(100, 1, 1)
        #     task.setKV(20, 1, 1)
        #     task.setActive(True, False, False)
        #     tasks.cloneAndAppend(task)
        # tasks.printToXML('motion_tracking_walking_cmc_tasks.xml')
        # TODO plotting should happen separately from generating the results.
        # cmc = osim.CMCTool()
        # cmc.setName('motion_tracking_walking_cmc')
        # cmc.setExternalLoadsFileName('grf_walk.xml')
        # # TODO filter:
        # cmc.setDesiredKinematicsFileName('coordinates.sto')
        # # cmc.setLowpassCutoffFrequency(6)
        # cmc.printToXML('motion_tracking_walking_cmc_setup.xml')
        cmc = osim.CMCTool('motion_tracking_walking_cmc_setup.xml')
        cmc.run()

        # TODO compare to MocoInverse.
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.01)
        inverse.set_kinematics_allow_extra_columns(True)
        # solution = inverse.solve()
        # solution.getMocoSolution().write(self.mocoinverse_solution_file)

    def report_results(self):
        fig = plt.figure(figsize=(4, 5.5))
        gs = gridspec.GridSpec(9, 2)
        sol_track = osim.MocoTrajectory(self.mocotrack_solution_file)
        time_track = sol_track.getTimeMat()
        pgc_track = 100.0 * (time_track - time_track[0]) / (
                    time_track[-1] - time_track[0])
        sol_inverse = osim.MocoTrajectory(self.mocoinverse_solution_file)
        time_inverse = sol_inverse.getTimeMat()
        pgc_inverse = 100.0 * (time_inverse - time_inverse[0]) / (
                    time_inverse[-1] - time_inverse[0])

        sol_cmc = osim.TimeSeriesTable('motion_tracking_walking_cmc_results/'
                                       'motion_tracking_walking_cmc_states.sto')
        time_cmc = np.array(sol_cmc.getIndependentColumn())
        pgc_cmc = 100.0 * (time_cmc - time_cmc[0]) / (time_cmc[-1] - time_cmc[0])
        def toarray(simtk_vector):
            array = np.empty(simtk_vector.size())
            for i in range(simtk_vector.size()):
                array[i] = simtk_vector[i]
            return array
        coords = [
            (
            f'/jointset/hip_{self.side}/hip_flexion_{self.side}', 'hip flexion',
            1.0),
            (f'/jointset/walker_knee_{self.side}/knee_angle_{self.side}',
             'knee flexion', 1.0),
            (f'/jointset/ankle_{self.side}/ankle_angle_{self.side}',
             'ankle plantarflexion', 1.0),
        ]
        for ic, coord in enumerate(coords):
            ax = plt.subplot(gs[(3 * ic):(3 * (ic + 1)), 0])
            y_track = coord[2] * np.rad2deg(
                sol_track.getStateMat(f'{coord[0]}/value'))
            ax.plot(pgc_track, y_track, label='track')
            y_inverse = coord[2] * np.rad2deg(
                sol_inverse.getStateMat(f'{coord[0]}/value'))
            ax.plot(pgc_inverse, y_inverse, label='inverse')
            y_cmc = coord[2] * np.rad2deg(
                toarray(sol_cmc.getDependentColumn(f'{coord[0]}/value')),)
            ax.plot(pgc_cmc, y_cmc, label='CMC')
            ax.set_xlim(0, 100)
            if ic == 0:
                ax.legend(frameon=False)
            if ic < len(coords) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            ax.set_ylabel(f'{coord[1]} (degrees)')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position('zero')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

        # TODO: Compare to EMG.
        muscles = [
            ('glut_max2', 'glutes'),
            ('psoas', 'iliopsoas'),
            ('semimem', 'hamstrings'),
            ('rect_fem', 'rectus femoris'),
            ('bifemsh', 'biceps femoris short head'),
            ('vas_int', 'vasti'),
            ('med_gas', 'gastrocnemius'),
            ('soleus', 'soleus'),
            ('tib_ant', 'tibialis anterior'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[im, 1])
            # TODO: percent gait cycle.
            activation_path = f'/forceset/{muscle[0]}_{self.side}/activation'
            ax.plot(pgc_track, sol_track.getStateMat(activation_path),
                    label='MocoTrack')
            ax.plot(pgc_inverse, sol_inverse.getStateMat(activation_path),
                    label='MocoInverse')
            ax.plot(pgc_cmc, toarray(sol_cmc.getDependentColumn(activation_path)))
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            if im < len(muscles) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            ax.set_title(muscle[1], fontsize=8)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

        fig.tight_layout(h_pad=0.05)

        fig.savefig('motion_tracking_walking.eps')
        fig.savefig('motion_tracking_walking.pdf')
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
