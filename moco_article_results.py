from abc import ABC, abstractmethod
import os
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from matplotlib_scalebar.scalebar import ScaleBar
import pylab as pl

import opensim as osim

# TODO: show icons of different muscles next to the activation plots.

mpl.rcParams.update({'font.size': 8,
                     'axes.titlesize': 8,
                     'axes.labelsize': 8,
                     'font.sans-serif': ['Arial']})

def publication_spines(axes):
    axes.spines['right'].set_visible(False)
    axes.yaxis.set_ticks_position('left')
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')

def toarray(simtk_vector):
    array = np.empty(simtk_vector.size())
    for i in range(simtk_vector.size()):
        array[i] = simtk_vector[i]
    return array

def nearest_index(array, val):
    return np.abs(array - val).argmin()

def shift_data_to_cycle(
        arbitrary_cycle_start_time, arbitrary_cycle_end_time,
        new_cycle_start_time, time, ordinate, cut_off=True):
    """
    Takes data (ordinate) that is (1) a function of time and (2) cyclic, and
    returns data that can be plotted so that the data starts at the desired
    part of the cycle.
    Used to shift data to the desired part of a gait cycle, for plotting
    purposes.  Data may be recorded from an arbitrary part
    of the gait cycle, but we might desire to plot the data starting at a
    particular part of the gait cycle (e.g., right foot strike).
    Another example use case is that one might have data for both right and
    left limbs, but wish to plot them together, and thus must shift data for
    one of the limbs by 50% of the gait cycle.
    This method also cuts the data so that your data covers at most a full gait
    cycle but not more.
    The first three parameters below not need exactly match times in the `time`
    array.
    This method can also be used just to truncate data, by setting
    `new_cycle_start_time` to be the same as `arbitrary_cycle_start_time`.
    Parameters
    ----------
    arbitrary_cycle_start_time : float
        Choose a complete cycle/period from the original data that you want to
        use in the resulting data. What is the initial time in this period?
    arbitrary_cycle_end_time : float
        See above; what is the final time in this period?
    new_cycle_start_time : float
        The time at which the shifted data should start. Note that the initial
        time in the shifted time array will regardlessly be 0.0, not
        new_cycle_start_time.
    time : np.array
        An array of times that must correspond with ordinate values (see next),
        and must contain arbitrary_cycle_start_time and
        arbitrary_cycle_end_time.
    ordinate : np.array
        The cyclic function of time, values corresponding to the times given.
    cut_off : bool, optional
        Sometimes, there's a discontinuity in the data that prevents obtaining
        a smooth curve if the data wraps around. In order prevent
        misrepresenting the data in plots, etc., an np.nan is placed in the
        appropriate place in the data.
    Returns
    -------
    shifted_time : np.array
        Same size as time parameter above, but its initial value is 0 and its
        final value is the duration of the cycle (arbitrary_cycle_end_time -
        arbitrary_cycle_start_time).
    shifted_ordinate : np.array
        Same ordinate values as before, but they are shifted so that the first
        value is ordinate[{index of arbitrary_cycle_start_time}] and the last
        value is ordinate[{index of arbitrary_cycle_start_time} - 1].
    Examples
    --------
    Observe that we do not require a constant interval for the time:
        >>> ordinate = np.array([2, 1., 2., 3., 4., 5., 6.])
        >>> time = np.array([0.5, 1.0, 1.2, 1.35, 1.4, 1.5, 1.8])
        >>> arbitrary_cycle_start_time = 1.0
        >>> arbitrary_cycle_end_time = 1.5
        >>> new_cycle_start_time = 1.35
        >>> shifted_time, shifted_ordinate = shift_data_to_cycle(
                ...     arbitrary_cycle_start_time, arbitrary_cycle_end_time,
                ...     new_cycle_start_time,
                ...     time, ordinate)
        >>> shifted_time
        array([ 0.  ,  0.05,  0.15,  0.3 ,  0.5 ])
        >>> shifted_ordinate
        array([3., 4., nan, 1., 2.])
    In order to ensure the entire duration of the cycle is kept the same,
    the time interval between the original times "1.5" and "1.0" is 0.1, which
    is the time gap between the original times "1.2" and "1.3"; the time
    between 1.2 and 1.3 is lost, and so we retain it in the place where we
    introduce a new gap (between "1.5" and "1.0"). NOTE that we only ensure the
    entire duration of the cycle is kept the same IF the available data covers
    the entire time interval [arbitrary_cycle_start_time,
    arbitrary_cycle_end_time].
    """
    # TODO gaps in time can only be after or before the time interval of the
    # available data.

    if new_cycle_start_time > arbitrary_cycle_end_time:
        raise Exception('(`new_cycle_start_time` = %f) > (`arbitrary_cycle_end'
                '_time` = %f), but we require that `new_cycle_start_time <= '
                '`arbitrary_cycle_end_time`.' % (new_cycle_start_time,
                    arbitrary_cycle_end_time))
    if new_cycle_start_time < arbitrary_cycle_start_time:
        raise Exception('(`new_cycle_start_time` = %f) < (`arbitrary_cycle'
                '_start_time` = %f), but we require that `new_cycle_start_'
                'time >= `arbitrary_cycle_start_time`.' % (new_cycle_start_time,
                    arbitrary_cycle_start_time))


    # We're going to modify the data.
    time = copy.deepcopy(time)
    ordinate = copy.deepcopy(ordinate)

    duration = arbitrary_cycle_end_time - arbitrary_cycle_end_time

    old_start_index = nearest_index(time, arbitrary_cycle_start_time)
    old_end_index = nearest_index(time, arbitrary_cycle_end_time)

    new_start_index = nearest_index(time, new_cycle_start_time)

    # So that the result matches exactly with the user's desired times.
    if new_cycle_start_time > time[0] and new_cycle_start_time < time[-1]:
        ordinate[new_start_index] = np.interp(new_cycle_start_time, time,
                ordinate)
        time[new_start_index] = new_cycle_start_time

    data_exists_before_arbitrary_start = old_start_index != 0
    if data_exists_before_arbitrary_start:
        #or (old_start_index == 0 and
        #    time[old_start_index] > arbitrary_cycle_start_time):
        # There's data before the arbitrary start.
        # Then we can interpolate to get what the ordinate SHOULD be exactly at
        # the arbitrary start.
        time[old_start_index] = arbitrary_cycle_start_time
        ordinate[old_start_index] = np.interp(arbitrary_cycle_start_time, time,
                ordinate)
        gap_before_avail_data = 0.0
    else:
        if not new_cycle_start_time < time[old_start_index]:
            gap_before_avail_data = (time[old_start_index] -
                    arbitrary_cycle_start_time)
        else:
            gap_before_avail_data = 0.0
    data_exists_after_arbitrary_end = time[-1] > arbitrary_cycle_end_time
    # TODO previous: old_end_index != (len(time) - 1)
    if data_exists_after_arbitrary_end:
        #or (old_end_index == (len(time) - 1)
        #and time[old_end_index] < arbitrary_cycle_end_time):
        time[old_end_index] = arbitrary_cycle_end_time
        ordinate[old_end_index] = np.interp(arbitrary_cycle_end_time, time,
                ordinate)
        gap_after_avail_data = 0
    else:
        gap_after_avail_data = arbitrary_cycle_end_time - time[old_end_index]

    # If the new cycle time sits outside of the available data, our job is much
    # easier; just add or subtract a constant from the given time.
    if new_cycle_start_time > time[-1]:
        time_at_end = arbitrary_cycle_end_time - new_cycle_start_time
        missing_time_at_beginning = \
                max(0, time[0] - arbitrary_cycle_start_time)
        move_forward = time_at_end + missing_time_at_beginning
        shift_to_zero = time[old_start_index:] - time[old_start_index]
        shifted_time = shift_to_zero + move_forward
        shifted_ordinate = ordinate[old_start_index:]
    elif new_cycle_start_time < time[0]:
        move_forward = time[0] - new_cycle_start_time
        shift_to_zero = time[:old_end_index + 1] - time[old_start_index]
        shifted_time = shift_to_zero + move_forward
        shifted_ordinate = ordinate[:old_end_index + 1]
    else:
        # We actually must cut up the data and move it around.

        # Interval of time in
        # [arbitrary_cycle_start_time, arbitrary_cycle_end_time] that is 'lost' in
        # doing the shifting.
        if new_cycle_start_time < time[old_start_index]:
            lost_time_gap = 0.0
        else:
            lost_time_gap = time[new_start_index] - time[new_start_index - 1]

        # Starts at 0.0.
        if new_cycle_start_time < time[0]:
            addin = gap_before_avail_data
        else:
            addin = 0
        first_portion_of_new_time = (time[new_start_index:old_end_index+1] -
                new_cycle_start_time + addin)

        # Second portion: (1) shift to 0, then move to the right of first portion.
        second_portion_to_zero = \
                time[old_start_index:new_start_index] - arbitrary_cycle_start_time
        second_portion_of_new_time = (second_portion_to_zero +
                first_portion_of_new_time[-1] + lost_time_gap +
                gap_after_avail_data)

        shifted_time = np.concatenate(
                (first_portion_of_new_time, second_portion_of_new_time))

        # Apply cut-off:
        if cut_off:
            ordinate[old_end_index] = np.nan

        # Shift the ordinate.
        shifted_ordinate = np.concatenate(
                (ordinate[new_start_index:old_end_index+1],
                    ordinate[old_start_index:new_start_index]))

    return shifted_time, shifted_ordinate

def plot_joint_moment_breakdown(model, moco_traj,
                                coord_paths, muscle_paths):
    model.initSystem()

    num_coords = len(coord_paths)
    num_muscles = len(muscle_paths)

    net_joint_moments = None
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        id_tool = osim.InverseDynamicsTool()
        id_tool.setModel(model)
        table = moco_traj.exportToStatesTable()
        labels = list(table.getColumnLabels())
        import re
        for ilabel in range(len(labels)):
            labels[ilabel] = labels[ilabel].replace('/value', '')
            labels[ilabel] = re.sub('/jointset/(.*?)/', '', labels[ilabel])
        table.setColumnLabels(labels)
        storage = osim.convertTableToStorage(table)
        # TODO: There's a bug in converting column labels in
        # convertTableToStorage().
        stolabels = osim.ArrayStr()
        stolabels.append('time')
        for label in labels:
            stolabels.append(label)
        storage.setColumnLabels(stolabels)
        id_tool.setCoordinateValues(storage)
        id_result = 'joint_moment_breakdown_residuals.sto'
        id_tool.setResultsDir(tmpdirname)
        id_tool.setOutputGenForceFileName(id_result)
        # TODO: Remove muscles from the model?
        id_tool.run()

        net_joint_moments = osim.TimeSeriesTable(os.path.join(tmpdirname, id_result))

    time = moco_traj.getTimeMat()

    states_traj = moco_traj.exportToStatesTrajectory(model)

    # TODO for models without activation dynamics, we must prescribeControlsToModel().

    fig = pl.figure(figsize=(8.5, 11))
    tendon_forces = np.empty((len(time), num_muscles))
    for imusc, muscle_path in enumerate(muscle_paths):
        muscle = model.getComponent(muscle_path)
        for itime in range(len(time)):
            state = states_traj.get(itime)
            model.realizeDynamics(state)
            tendon_forces[itime, imusc] = muscle.getTendonForce(state)

    for icoord, coord_path in enumerate(coord_paths):
        coord = model.getComponent(coord_path)

        label = os.path.split(coord_path)[-1] + '_moment'
        net_moment = toarray(net_joint_moments.getDependentColumn(label))

        moment_arms = np.empty((len(time), num_muscles))
        for imusc, muscle_path in enumerate(muscle_paths):
            muscle = model.getComponent(muscle_path)
            for itime in range(len(time)):
                state = states_traj.get(itime)
                moment_arms[itime, imusc] = \
                    muscle.computeMomentArm(state, coord)

        ax = fig.add_subplot(num_coords, 2, 2 * icoord + 1)
        net_integ = np.trapz(np.abs(net_moment), x=time)
        sum_actuators_shown = np.zeros_like(time)
        for imusc, muscle_path in enumerate(muscle_paths):
            if np.any(moment_arms[:, imusc]) > 0.00001:
                this_moment = tendon_forces[:, imusc] * moment_arms[:, imusc]
                mom_integ = np.trapz(np.abs(this_moment), time)
                if mom_integ > 0.05 * net_integ:
                    ax.plot(time, this_moment, label=muscle_path)

                    sum_actuators_shown += this_moment

        ax.plot(time, sum_actuators_shown,
                label='sum actuators shown', color='gray', linewidth=2)

        ax.plot(time, net_moment,
                label='net', color='black', linewidth=2)

        ax.set_title(coord_path)
        ax.set_ylabel('moment (N-m)')
        ax.legend(frameon=False, bbox_to_anchor=(1, 1),
                loc='upper left', ncol=2)
        ax.tick_params(axis='both')
    ax.set_xlabel('time (% gait cycle)')

    fig.tight_layout()
    return fig

class MocoPaperResult(ABC):
    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass


class SuspendedMass(MocoPaperResult):
    width = 0.2

    def __init__(self):
        pass
    def build_model(self):
        model = osim.ModelFactory.createPlanarPointMass()
        body = model.updBodySet().get("body")
        model.updForceSet().clearAndDestroy()
        model.finalizeFromProperties()

        actuL = osim.DeGrooteFregly2016Muscle()
        actuL.setName("left")
        actuL.set_max_isometric_force(10)
        actuL.set_optimal_fiber_length(.20)
        actuL.set_tendon_slack_length(0.10)
        actuL.set_pennation_angle_at_optimal(0.0)
        actuL.set_ignore_tendon_compliance(True)
        actuL.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(-self.width, 0, 0))
        actuL.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuL)

        actuM = osim.DeGrooteFregly2016Muscle()
        actuM.setName("middle")
        actuM.set_max_isometric_force(10)
        actuM.set_optimal_fiber_length(0.09)
        actuM.set_tendon_slack_length(0.1)
        actuM.set_pennation_angle_at_optimal(0.0)
        actuM.set_ignore_tendon_compliance(True)
        actuM.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(0, 0, 0))
        actuM.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuM)

        actuR = osim.DeGrooteFregly2016Muscle()
        actuR.setName("right")
        actuR.set_max_isometric_force(20)
        actuR.set_optimal_fiber_length(.21)
        actuR.set_tendon_slack_length(0.09)
        actuR.set_pennation_angle_at_optimal(0.0)
        actuR.set_ignore_tendon_compliance(True)
        actuR.addNewPathPoint("origin", model.updGround(),
                              osim.Vec3(+self.width, 0, 0))
        actuR.addNewPathPoint("insertion", body, osim.Vec3(0))
        model.addForce(actuR)

        model.finalizeConnections()
        return model

    def create_study(self):
        study = osim.MocoStudy()
        problem = study.updProblem()
        problem.setModel(self.build_model())
        problem.setTimeBounds(0, 0.5)
        problem.setStateInfo("/jointset/tx/tx/value", [-0.03, 0.03], -0.03,
                             0.03)
        problem.setStateInfo("/jointset/ty/ty/value", [-2 * self.width, 0],
                             -self.width,
                             -self.width + 0.05)
        problem.setStateInfo("/jointset/tx/tx/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/jointset/ty/ty/speed", [-15, 15], 0, 0)
        problem.setStateInfo("/forceset/left/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/middle/activation", [0, 1], 0)
        problem.setStateInfo("/forceset/right/activation", [0, 1], 0)
        problem.setControlInfo("/forceset/left", [0, 1])
        problem.setControlInfo("/forceset/middle", [0, 1])
        problem.setControlInfo("/forceset/right", [0, 1])

        solver = study.initCasADiSolver()
        solver.set_num_mesh_points(101)

        return study

    def predict(self):

        study = self.create_study()
        problem = study.updProblem()

        problem.addGoal(osim.MocoControlGoal())
        solution = study.solve()

        # study.visualize(solution)

        time_stepping = \
            osim.simulateIterateWithTimeStepping(solution, self.build_model())

        return solution, time_stepping

    def track(self, prediction_solution, exponent=2):

        study = self.create_study()
        problem = study.updProblem()

        tracking = osim.MocoStateTrackingGoal("tracking", 10.0)
        tracking.setReference(
            osim.TableProcessor(prediction_solution.exportToStatesTable()))
        tracking.setPattern('.*value$')
        tracking.setAllowUnusedReferences(True)
        problem.addGoal(tracking)
        effort = osim.MocoControlGoal("effort")
        effort.setExponent(exponent)
        problem.addGoal(effort)

        solution = study.solve()

        return solution

    def generate_results(self):
        predict_solution, time_stepping = self.predict()
        predict_solution.write('results/suspended_mass_prediction_solution.sto')
        time_stepping.write('results/suspended_mass_time_stepping.sto')
        track_solution = self.track(predict_solution)
        track_solution.write('results/suspended_mass_track_solution.sto')
        track_solution_p = self.track(predict_solution, 5)
        track_solution_p.write('results/suspended_mass_track_p_solution.sto')

    def report_results(self):
        pl.figure()
        fig = plt.figure(figsize=(5.5, 3))
        grid = gridspec.GridSpec(3, 2)
        predict_solution = osim.MocoTrajectory(
            'results/suspended_mass_prediction_solution.sto')
        time_stepping = osim.MocoTrajectory(
            'results/suspended_mass_time_stepping.sto')
        track_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_solution.sto')
        track_p_solution = osim.MocoTrajectory(
            'results/suspended_mass_track_p_solution.sto')

        ax = fig.add_subplot(grid[0:3, 0])
        ax.plot(predict_solution.getStateMat('/jointset/tx/tx/value'),
                predict_solution.getStateMat('/jointset/ty/ty/value'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getStateMat('/jointset/tx/tx/value'),
                time_stepping.getStateMat('/jointset/ty/ty/value'),
                linestyle='--',
                color='red')
        ax.plot(track_solution.getStateMat('/jointset/tx/tx/value'),
                track_solution.getStateMat('/jointset/ty/ty/value'),
                linestyle=':',
                color='blue')
        ax.plot(track_p_solution.getStateMat('/jointset/tx/tx/value'),
                track_p_solution.getStateMat('/jointset/ty/ty/value'),
                linestyle=':',
                color='green')
        scalebar = AnchoredSizeBar(ax.transData, 0.01, label='1 cm',
                                   # loc=(0, 0.3),
                                   loc = 'center left',
                                   pad=0.5, frameon=False)
        ax.add_artist(scalebar)
        ax.set_title('trajectory')
        # ax.set_ylabel('y (m)')
        # ax.set_xlabel('x (m)')
        # ax.set_yticks([-0.20, -0.18, -0.16])
        # ax.set_xticks([-0.02, 0, 0.02])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.legend(['prediction', 'time stepping', 'tracking', 'tracking p=5'],
                  frameon=False,
                  handlelength=1.9)
        publication_spines(ax)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax = fig.add_subplot(grid[0, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/left/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/left/activation'),
                linestyle='--',
                color='red')
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/left/activation'),
                linestyle=':',
                color='blue')
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/left/activation'),
                linestyle=':',
                color='green')
        ax.set_title('left activation')
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        publication_spines(ax)

        ax = fig.add_subplot(grid[1, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/middle/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/middle/activation'),
                linestyle='--',
                color='red')
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/middle/activation'),
                linestyle=':',
                color='blue')
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/middle/activation'),
                linestyle=':',
                color='green')
        ax.set_title('middle activation')
        ax.set_yticks([0, 1])
        ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        publication_spines(ax)

        ax = fig.add_subplot(grid[2, 1])
        ax.plot(predict_solution.getTimeMat(),
                predict_solution.getStateMat('/forceset/right/activation'),
                color='gray', linewidth=4)
        ax.plot(time_stepping.getTimeMat(),
                time_stepping.getStateMat('/forceset/right/activation'),
                linestyle='--',
                color='red')
        ax.plot(track_solution.getTimeMat(),
                track_solution.getStateMat('/forceset/right/activation'),
                linestyle=':',
                color='blue')
        ax.plot(track_p_solution.getTimeMat(),
                track_p_solution.getStateMat('/forceset/right/activation'),
                linestyle=':',
                color='green')
        ax.set_title('right activation')
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel('time (s)')
        publication_spines(ax)

        fig.tight_layout()

        # pl.legend(predict_solution.getControlNames())
        pl.savefig('figures/suspended_mass.png', dpi=600)
        pl.savefig('figures/suspended_mass.eps')
        pl.savefig('figures/suspended_mass.pdf')

    # TODO surround the point with muscles and maximize distance traveled.


class MotionTrackingWalking(MocoPaperResult):
    def __init__(self):
        self.initial_time = 0.450 # 0.73
        self.final_time = 1.565 # 1.795
        self.footstrike = 1.424
        self.mocotrack_solution_file = \
            'results/motion_tracking_walking_track_solution.sto'
        self.mocoinverse_solution_file = \
            'results/motion_tracking_walking_inverse_solution.sto'
        self.side = 'r'

    def shift(self, time, y):
        return shift_data_to_cycle(self.initial_time, self.final_time,
                                   self.footstrike, time, y)
        # time = copy.deepcopy(time)
        # y = copy.deepcopy(y)
        # # pgc = np.linspace(0, 100, 401)
        # index_before_strike = np.arange(len(time))[time < self.footstrike][-1]
        # index_after_strike = index_before_strike + 1
        # assert index_after_strike < len(time) - 1
        #
        # # TODO: want the greatest index less than self.footstrike.
        # y[index_after_strike] = np.interp(self.footstrike, time, y)
        # time[index_after_strike] = self.footstrike
        #
        # first_portion_of_new_time = time[index_after_strike::] - self.footstrike
        # gap_before_footstrike = self.footstrike - time[index_before_strike]
        # second_portion_of_new_time = (time[0:index_after_strike]
        #                               - time[0] + first_portion_of_new_time[-1]
        #                               + gap_before_footstrike)
        # # print('DEBUG', first_portion_of_new_time, second_portion_of_new_time)
        # shifted_time = np.concatenate(first_portion_of_new_time,
        #                               second_portion_of_new_time)
        # shifted_y = np.concatenate(y[index_after_strike::],
        #                            y[0:index_after_strike])
        # # TODO: final_time - initial_time?
        # assert shifted_time[-1] == time[-1] - time[0]
        # assert len(shifted_time) == len(time)
        # assert len(shifted_y) == len(y)
        # return shifted_time, shifted_y



    def create_model_processor(self):
        modelProcessor = osim.ModelProcessor(
            # "resources/ArnoldSubject02Walk3/subject02_armless.osim")
            "resources/Rajagopal2016/subject_walk_armless.osim")
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(
            ['subtalar_r', 'mtp_r', 'subtalar_l', 'mtp_l']))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
        # modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
        modelProcessor.append(osim.ModOpAddReserves(1))
        modelProcessor.process().printToXML("subject_armless_for_cmc.osim")
        # ext_loads_xml = "resources/ArnoldSubject02Walk3/external_loads.xml"
        ext_loads_xml = "resources/Rajagopal2016/grf_walk.xml"
        modelProcessor.append(osim.ModOpAddExternalLoads(ext_loads_xml))
        return modelProcessor

    def generate_results(self):
        # Create and name an instance of the MocoTrack tool.
        track = osim.MocoTrack()
        track.setName("motion_tracking_walking")

        modelProcessor = self.create_model_processor()
        track.setModel(modelProcessor)

        # TODO:
        #  - avoid removing muscle passive forces
        #  - play with weights between tracking and effort.
        #  - report duration to solve the problem.
        #  - figure could contain still frames of the model throughout the motion.
        #  - make sure residuals and reserves are small (generate_report).
        #  - try using Millard muscle.
        #  - plot joint moment breakdown.

        coordinates = osim.TableProcessor(
            # "resources/ArnoldSubject02Walk3/subject02_walk3_ik_solution.mot")
            "resources/Rajagopal2016/coordinates.sto")
        coordinates.append(osim.TabOpLowPassFilter(6))
        track.setStatesReference(coordinates)
        # track.set_states_global_tracking_weight(10)

        # This setting allows extra data columns contained in the states
        # reference that don't correspond to model coordinates.
        track.set_allow_unused_references(True)

        track.set_track_reference_position_derivatives(True)

        # Initial time, final time, and mesh interval.
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(0.01)

        moco = track.initialize()
        moco.set_write_solution("results/")

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

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.set_optim_convergence_tolerance(1e-4)

        # Solve and visualize.
        moco.printToXML('motion_tracking_walking.omoco')
        # 45 minutes
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
        # 1 minute
        cmc.run()

        # TODO compare to MocoInverse.
        inverse = osim.MocoInverse()
        inverse.setModel(modelProcessor)
        inverse.setKinematics(coordinates)
        inverse.set_initial_time(self.initial_time)
        inverse.set_final_time(self.final_time)
        inverse.set_mesh_interval(0.01)
        inverse.set_kinematics_allow_extra_columns(True)
        inverse.set_tolerance(1e-5)
        # 2 minutes
        solution = inverse.solve()
        solution.getMocoSolution().write(self.mocoinverse_solution_file)

        # TODO: Minimize joint reaction load!
    def plot(self, ax, time, y, *args, **kwargs):
        shifted_time, shifted_y = self.shift(time, y)
        # TODO is this correct?
        duration = self.final_time - self.initial_time
        ax.plot(100.0 * shifted_time / duration, shifted_y, *args, **kwargs)

    def report_results(self):
        sol_track = osim.MocoTrajectory(self.mocotrack_solution_file)
        time_track = sol_track.getTimeMat()
        sol_inverse = osim.MocoTrajectory(self.mocoinverse_solution_file)
        time_inverse = sol_inverse.getTimeMat()

        modelProcessor = self.create_model_processor()
        model = modelProcessor.process()
        report = osim.report.Report(model, self.mocoinverse_solution_file)
        report.generate()

        # TODO: slight shift in CMC solution might be due to how we treat
        # percent gait cycle and the fact that CMC is missing 0.02 seconds.
        sol_cmc = osim.TimeSeriesTable('results/motion_tracking_walking_cmc_results/'
                                       'motion_tracking_walking_cmc_states.sto')
        # TODO uh oh overwriting the MocoInverse solution.
        # sol_inverse.setStatesTrajectory(sol_cmc)
        time_cmc = np.array(sol_cmc.getIndependentColumn())

        fig = plot_joint_moment_breakdown(model, sol_inverse,
                                    ['/jointset/hip_r/hip_flexion_r',
                                     '/jointset/walker_knee_r/knee_angle_r',
                                     '/jointset/ankle_r/ankle_angle_r'],
                                    ['/forceset/glmax2_r',
                                     '/forceset/psoas_r',
                                     '/forceset/semimem_r',
                                     '/forceset/recfem_r',
                                     '/forceset/bfsh_r',
                                     '/forceset/vasint_r',
                                     '/forceset/gasmed_r',
                                     '/forceset/soleus_r',
                                     '/forceset/tibant_r'])
                                    # ['/forceset/glut_max2_r',
                                    #  '/forceset/psoas_r',
                                    #  '/forceset/semimem_r',
                                    #  '/forceset/rect_fem_r',
                                    #  '/forceset/bifemsh_r',
                                    #  '/forceset/vas_int_r',
                                    #  '/forceset/med_gas_r',
                                    #  '/forceset/soleus_r',
                                    #  '/forceset/tib_ant_r'])
        fig.savefig('joint_moment_breakdown')
        # fig.show()

        fig = plt.figure(figsize=(5.5, 5.5))
        gs = gridspec.GridSpec(9, 2)


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

            y_cmc = coord[2] * np.rad2deg(
                toarray(sol_cmc.getDependentColumn(f'{coord[0]}/value')),)
            self.plot(ax, time_cmc, y_cmc, label='CMC', color='gray')

            y_inverse = coord[2] * np.rad2deg(
                sol_inverse.getStateMat(f'{coord[0]}/value'))
            self.plot(ax, time_inverse, y_inverse, label='MocoInverse',
                      color='k', linestyle='--')

            # y_track = coord[2] * np.rad2deg(
            #     sol_track.getStateMat(f'{coord[0]}/value'))
            # self.plot(ax, time_track, y_track, label='MocoTrack', color='blue', linestyle='--')

            ax.set_xlim(0, 100)
            if ic == 1:
                ax.legend(frameon=False, handlelength=1.9)
            if ic < len(coords) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
                ax.get_xaxis().set_label_coords(0.5, 0)
            ax.set_ylabel(f'{coord[1]} (degrees)')
            ax.get_yaxis().set_label_coords(-0.15, 0.5)

            ax.spines['bottom'].set_position('zero')
            publication_spines(ax)

        # TODO: Compare to EMG.
        muscles = [
            ('glmax2', 'gluteus maximus'),
            ('psoas', 'iliopsoas'),
            ('semimem', 'hamstrings'),
            ('recfem', 'rectus femoris'),
            ('bfsh', 'biceps femoris short head'),
            ('vasint', 'vasti'),
            ('gasmed', 'gastrocnemius'),
            ('soleus', 'soleus'),
            ('tibant', 'tibialis anterior'),
            # ('glut_max2', 'gluteus maximus'),
            # ('psoas', 'iliopsoas'),
            # ('semimem', 'hamstrings'),
            # ('rect_fem', 'rectus femoris'),
            # ('bifemsh', 'biceps femoris short head'),
            # ('vas_int', 'vasti'),
            # ('med_gas', 'gastrocnemius'),
            # ('soleus', 'soleus'),
            # ('tib_ant', 'tibialis anterior'),
        ]
        for im, muscle in enumerate(muscles):
            ax = plt.subplot(gs[im, 1])
            activation_path = f'/forceset/{muscle[0]}_{self.side}/activation'
            self.plot(ax, time_cmc, toarray(sol_cmc.getDependentColumn(activation_path)),
                    color='gray')
            self.plot(ax, time_inverse, sol_inverse.getStateMat(activation_path),
                    label='Inverse',
                    color='k', linestyle='--')
            # self.plot(ax, time_track, sol_track.getStateMat(activation_path),
            #         label='Track',
            #         color='blue', linestyle='--')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            if im < len(muscles) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (% gait cycle)')
            ax.set_title(f'{muscle[1]} activation')

            publication_spines(ax)

        fig.tight_layout(h_pad=0.05)

        fig.savefig('figures/motion_tracking_walking.eps')
        fig.savefig('figures/motion_tracking_walking.pdf')
        fig.savefig('figures/motion_tracking_walking.png', dpi=600)

class CrouchToStand(MocoPaperResult):
    def __init__(self):
        self.predict_solution_file = 'predict_solution.sto'
        self.predict_assisted_solution_file = 'predict_assisted_solution.sto'
        self.track_solution_file = 'track_solution.sto'
        self.inverse_solution_file = 'inverse_solution.sto'
        self.inverse_assisted_solution_file = 'inverse_assisted_solution.sto'

    def create_study(self, model):
        moco = osim.MocoStudy()
        moco.set_write_solution("results/")
        solver = moco.initCasADiSolver()
        solver.set_dynamics_mode('implicit')
        # TODO: More mesh points.
        solver.set_num_mesh_points(50)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_optim_constraint_tolerance(1e-3)
        solver.set_optim_finite_difference_scheme('forward')

        problem = moco.updProblem()
        problem.setModelCopy(model)
        problem.setTimeBounds(0, [0.1, 2])
        # problem.setTimeBounds(0, 1)
        problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value',
                          [-2, 0.5], -2, 0)
        problem.setStateInfo('/jointset/knee_r/knee_angle_r/value',
                          [-2, 0], -2, 0)
        problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value',
                          [-0.5, 0.7], -0.5, 0)
        problem.setStateInfoPattern('/jointset/.*/speed', [], 0, 0)
        # TODO: Minimize initial activation.

        # for muscle in model.getMuscles():
        #     if not muscle.get_ignore_activation_dynamics():
        #         muscle_path = muscle.getAbsolutePathString()
        #         problem.setStateInfo(muscle_path + '/activation', [0, 1], 0)
        #         problem.setControlInfo(muscle_path, [0, 1], 0)
        return moco

    def muscle_driven_model(self):
        model = osim.Model('resources/sitToStand_3dof9musc.osim')
        model.finalizeConnections()
        osim.DeGrooteFregly2016Muscle.replaceMuscles(model)
        for muscle in model.getMuscles():
            muscle.set_ignore_tendon_compliance(True)
            muscle.set_max_isometric_force(2 * muscle.get_max_isometric_force())
            dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
            # dgf.set_active_force_width_scale(1.5)
            if muscle.getName() == 'soleus_r':
                dgf.set_ignore_passive_fiber_force(True)
        return model

    def predict(self):
        moco = self.create_study(self.muscle_driven_model())
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))
        problem.addGoal(osim.MocoInitialActivationGoal('init_activation'))
        # TODO: Weird goal to have:
        problem.addGoal(osim.MocoFinalTimeGoal('time', 0.1))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        solution = moco.solve()
        solution.write(self.predict_solution_file)

        # TODO: Why does TA turn on?
        # TODO: Make sure we have a global optimum. Use random initial guesses.
        # TODO: Sweep through final times and see which gives the minimum cost.
        # TODO: Does not track well!
        # timeStepSolution = osim.simulateIterateWithTimeStepping(
        #     predictSolution, problem.getPhase(0).getModel(),
        #     1e-8)
        # timeStepSolution.write('timeStepSolution.sto')
        # moco.visualize(timeStepSolution)

        return solution

    def predict_assisted(self):
        model = self.muscle_driven_model()
        device = osim.SpringGeneralizedForce('knee_angle_r')
        device.setName('spring')
        device.setStiffness(50)
        device.setRestLength(0)
        device.setViscosity(0)
        model.addForce(device)

        moco = self.create_study(model)
        problem = moco.updProblem()
        problem.addGoal(osim.MocoControlGoal('effort'))

        problem.addParameter(
           osim.MocoParameter('stiffness', '/forceset/spring',
                              'stiffness', osim.MocoBounds(0, 80)))

        solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver())
        solver.resetProblem(problem)

        solver.set_parameters_require_initsystem(False)
        solution = moco.solve()
        moco.visualize(solution)
        solution.write(self.predict_assisted_solution_file)

    def generate_results(self):
        self.predict()
        self.predict_assisted()

    def report_results(self):
        fig = plt.figure(figsize=(5.5, 5.5))
        values = [
            '/jointset/hip_r/hip_flexion_r/value',
            '/jointset/knee_r/knee_angle_r/value',
            '/jointset/ankle_r/ankle_angle_r/value',
        ]
        coord_names = {
            '/jointset/hip_r/hip_flexion_r/value': 'hip flexion',
            '/jointset/knee_r/knee_angle_r/value': 'knee flexion',
            '/jointset/ankle_r/ankle_angle_r/value': 'ankle dorsiflexion',
        }
        coord_signs = {
            '/jointset/hip_r/hip_flexion_r/value': -1.0,
            '/jointset/knee_r/knee_angle_r/value': -1.0,
            '/jointset/ankle_r/ankle_angle_r/value': -1.0,
        }

        muscles = [
            ('glut_max2', 'gluteus maximus'),
            ('psoas', 'iliopsoas'),
            ('semimem', 'hamstrings'),
            ('rect_fem', 'rectus femoris'),
            ('bifemsh', 'biceps femoris short head'),
            ('vas_int', 'vasti'),
            ('med_gas', 'gastrocnemius'),
            ('soleus', 'soleus'),
            ('tib_ant', 'tibialis anterior'),
        ]
        # grid = plt.GridSpec(9, 2, hspace=0.7,
        #                     left=0.1, right=0.98, bottom=0.07, top=0.96,
        #                     )
        grid = gridspec.GridSpec(9, 2)
        coord_axes = []
        for ic, coordvalue in enumerate(values):
            ax = fig.add_subplot(grid[3 * ic: 3 * (ic + 1), 0])
            ax.set_ylabel('%s (degrees)' % coord_names[coordvalue])
            ax.get_yaxis().set_label_coords(-0.15, 0.5)
            if ic == len(values) - 1:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            publication_spines(ax)
            ax.spines['bottom'].set_position('zero')
            coord_axes += [ax]
        muscle_axes = []
        for im, muscle in enumerate(muscles):
            ax = fig.add_subplot(grid[im, 1])
            ax.set_title('%s activation' % muscle[1])
            ax.set_ylim([0, 1])
            if im == len(muscles) - 1:
                ax.set_xlabel('time (s)')
            else:
                ax.set_xticklabels([])
            publication_spines(ax)
            muscle_axes += [ax]
        def plot_solution(sol, label, linestyle='-', color='k'):
            time = sol.getTimeMat()
            for ic, coordvalue in enumerate(values):
                ax = coord_axes[ic]
                if ic == 0:
                    use_label = label
                else:
                    use_label = None
                if coordvalue in sol.getStateNames():
                    y = (coord_signs[coordvalue] * np.rad2deg(
                        sol.getStateMat(coordvalue)))
                    ax.plot(time, y, linestyle=linestyle, color=color,
                            label=use_label)
                    # ax.set_xlim(time[0], time[-1])
                else:
                    if ic == 0:
                        ax.plot(0, 0, label=use_label)
            for im, muscle in enumerate(muscles):
                ax = muscle_axes[im]
                ax.plot(time,
                        sol.getStateMat(
                            '/forceset/%s_r/activation' % muscle[0]),
                        linestyle=linestyle, color=color)
                # ax.set_xlim(time[0], time[-1])


        predict_solution = osim.MocoTrajectory(self.predict_solution_file)

        predict_assisted_solution = osim.MocoTrajectory(
            self.predict_assisted_solution_file)
        plot_solution(predict_solution, 'prediction', '-', 'k')
        plot_solution(predict_assisted_solution, 'prediction with assistance',
                      '--', 'blue')
        coord_axes[0].legend(frameon=False, handlelength=1.9)
        fig.tight_layout(h_pad=0.05)
        fig.savefig('figures/crouch_to_stand.png', dpi=600)


        # fig = plot_joint_moment_breakdown(self.muscle_driven_model(),
        #                                   predict_solution,
        #                             ['/jointset/hip_r/hip_flexion_r',
        #                              '/jointset/knee_r/knee_angle_r',
        #                              '/jointset/ankle_r/ankle_angle_r'],
        #                             ['/forceset/glut_max2_r',
        #                              '/forceset/psoas_r',
        #                              '/forceset/semimem_r',
        #                              '/forceset/rect_fem_r',
        #                              '/forceset/bifemsh_r',
        #                              '/forceset/vas_int_r',
        #                              '/forceset/med_gas_r',
        #                              '/forceset/soleus_r',
        #                              '/forceset/tib_ant_r'])
        # fig.show()
        # return # TODO

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate results for the"
                                                 "OpenSim Moco publication.")
    parser.add_argument('--no-generate', dest='generate', action='store_false',
                        help='Skip generating the results; only report.')
    parser.set_defaults(generate=True)
    args = parser.parse_args()

    results = [
        # SuspendedMass(),
        MotionTrackingWalking(),
        # MotionPredictionAndAssistanceWalking(),
        # CrouchToStand(),
        ]
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
