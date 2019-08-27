import os
import copy
import numpy as np
import pylab as pl

import opensim as osim

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
                                coord_paths, muscle_paths=None):
    model.initSystem()

    num_coords = len(coord_paths)

    if not muscle_paths:
        muscle_paths = list()
        for muscle in model.getMuscleList():
            muscle_paths.append(muscle.getAbsolutePathString())
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
                if mom_integ > 0.01 * net_integ:
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
