import os
import copy
import numpy as np
import pylab as pl
from scipy.signal import butter, filtfilt

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
                                coord_paths, muscle_paths=None,
                                coordact_paths=[]):
    model.initSystem()

    num_coords = len(coord_paths)

    if not muscle_paths:
        muscle_paths = list()
        for muscle in model.getMuscleList():
            muscle_paths.append(muscle.getAbsolutePathString())
    num_muscles = len(muscle_paths)

    num_coordact = len(coordact_paths)


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


    coordact_moments = np.empty((len(time), num_coordact))
    for ica, coordact_paths in enumerate(coordact_paths):
        coordact = model.getComponent(coordact_paths)
        for itime in range(len(time)):
            state = states_traj.get(itime)
            model.realizeDynamics(state)
            coordact_moments[itime, ica] = coordact.getActuation(state)


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

        for ica, coordact_path in enumerate(coordact_paths):
            this_moment = coordact_moments[:, ica]
            ax.plot(time, this_moment, label=coordact_path)
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

class ANCFile(object):
    """A plain-text file format for storing analog data from Motion Analysis
    Realtime. They have a file extension '.anc'. The file extension '.anb' is
    for binary files.
    The metadata for the file is stored in attributes of this object.
    This class is based off of similar code written by Amy Silder.
    """
    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            Valid file path to an ANC (.anc) file.
        """
        with open(fpath) as f:
            line1 = f.readline()
            line1list = line1.split('\t')
            self.file_type = line1list[1].strip()
            self.generation = line1list[3].strip()

            line2 = f.readline()
            line2list = line2.split('\t')
            self.board_type = line2list[1].strip()
            self.polarity = line2list[3].strip()

            line3 = f.readline()
            line3list = line3.split('\t')
            self.trial_name = line3list[1]
            self.trial_num = int(line3list[3])
            self.duration = float(line3list[5])
            self.num_channels = int(line3list[7])

            line4 = f.readline()
            line4list = line4.split('\t')
            self.bit_depth = int(line4list[1])
            self.precise_rate = float(line4list[3])

            line = f.readline()
            iline = 5
            while line.strip() == '':
                # There will most likely be a few empty lines.
                line = f.readline()
                iline += 1

            # Metadata for each column.
            header_row = line
            self.names = header_row.split()[1:]
            rate_row = f.readline()
            iline += 1
            self.rates = {self.names[i]: float(v) for i, v in
                    enumerate(rate_row.split()[1:])}
            range_row = f.readline()
            iline += 1
            self.ranges = {self.names[i]: float(v) for i, v in
                    enumerate(range_row.split()[1:])}

        dtype = {'names': ['time'] + self.names,
                'formats': (len(self.names) + 1) * ['float64']}
        self.data = np.loadtxt(fpath, delimiter='\t', skiprows=iline,
                    dtype=dtype)
        self.time = self.data['time']

    def __getitem__(self, name):
        """See `column()`.
        """
        return self.column(name)

    def __setitem__(self, name, val):
        """self.data[name] = val
        """
        self.data[name] = val

    def column(self, name):
        """
        Parameters
        ----------
        name : str
            Name of a column in the file (e.g., 'F1X'). For the 'time', column,
            just get the 'time' attribute.
        Returns
        -------
        col : np.array
            The data you are looking for.
        """
        return self.data[name]

def remove_fields_from_structured_ndarray(ndarray, fields, copy=True):
    """Returns the ndarray but now without the fields specified.
    Parameters
    ----------
    ndarray: numpy.ndarray
        The structured ndarray from which to remove a field (and corresponding
        data).
    fields: list of str's, or a single str.
        e.g., 'F2X'.
    copy: bool, optional
        NumPy doesn't like it if you tamper with the returned ndarray, since we
        used array indexing to give it to you. It prefers we return a copy
        of the array. This takes more time/memory, though. So if you are not
        going to tamper with the returned value, you may want `copy` to be
        False.
    Returns
    -------
    new_ndarray: numpy.ndarray
    """
    names = list(ndarray.dtype.names)
    if type(fields) != list:
        fields = [fields]
    for field in fields:
        names.remove(field)
    if copy:
        return ndarray[names].copy()
    else:
        return ndarray[names]

def ndarray2storage(ndarray, storage_fpath, name=None, in_degrees=False):
    """Saves an ndarray, with named dtypes, to an OpenSim Storage file.
    Parameters
    ----------
    ndarray : numpy.ndarray
    storage_fpath : str
    in_degrees : bool, optional
    name : str
        Name of Storage object.
    """
    n_rows = ndarray.shape[0]
    n_cols = len(ndarray.dtype.names)

    f = open(storage_fpath, 'w')
    f.write('%s\n' % (name if name else storage_fpath,))
    f.write('version=1\n')
    f.write('nRows=%i\n' % n_rows)
    f.write('nColumns=%i\n' % n_cols)
    f.write('inDegrees=%s\n' % ('yes' if in_degrees else 'no',))
    f.write('endheader\n')
    for line_num, col in enumerate(ndarray.dtype.names):
        if line_num != 0:
            f.write('\t')
        f.write('%s' % col)
    f.write('\n')

    for i_row in range(n_rows):
        for line_num, col in enumerate(ndarray.dtype.names):
            if line_num != 0:
                f.write('\t')
            f.write('%f' % ndarray[col][i_row])
        f.write('\n')

    f.close()

def filter_emg(raw_signal, sampling_rate, bandpass_order=6,
        bandpass_lower_frequency=50, bandpass_upper_frequency=500,
        lowpass_order=4,
        lowpass_frequency=7.5,
        cd_lowpass_frequency=15.0):
    """Filters a raw EMG signal. The signal must have been sampled at a
    constant rate. We perform the following steps:
    1. Butterworth bandpass filter.
    2. Butterworth lowpass filter.
    3. Critically damped lowpass filter.
    The signals are applied forward and backward (`filtfilt`), which should
    prevent a time delay.
    Parameters
    ----------
    raw_signal : array_like
        Raw EMG signal.
    sampling_rate : float
        In Hertz.
    bandpass_order : int, optional
    bandpass_lower_frequency : float, optional
        In the bandpass filter, what is the lower cutoff frequency? In Hertz.
    bandpass_upper_frequency : float, optional
        In the bandpass filter, what is the upper cutoff frequency? In Hertz.
    lowpass_order : int, optional
    lowpass_frequency : float, optional
        In the lowpass filter, what is the cutoff frequency? In Hertz.
    cd_lowpass_frequency : float, optional
        In the Critically damped lowpass filter, what is the cutoff frequency?
        In Hertz.
    Returns
    -------
    filtered_signal : array_like
    """
    nyquist_frequency = 0.5 * sampling_rate

    # Bandpass.
    # ---------
    normalized_bandpass_lower = bandpass_lower_frequency / nyquist_frequency
    normalized_bandpass_upper = bandpass_upper_frequency / nyquist_frequency
    bandpass_cutoffs = [normalized_bandpass_lower, normalized_bandpass_upper]
    bandpass_b, bandpass_a = butter(bandpass_order, bandpass_cutoffs,
            btype='bandpass')

    bandpassed = filtfilt(bandpass_b, bandpass_a, raw_signal)

    # Rectify.
    # --------
    rectified = np.abs(bandpassed)

    # Lowpass.
    # --------
    lowpass_cutoff = lowpass_frequency / nyquist_frequency
    lowpass_b, lowpass_a = butter(lowpass_order, lowpass_cutoff)

    lowpassed = filtfilt(lowpass_b, lowpass_a, rectified)

    # Critically damped filter.
    # -------------------------
    cd_order = 4
    cdfed = filter_critically_damped(lowpassed, sampling_rate,
            cd_lowpass_frequency, order=4)

    return cdfed

def filter_critically_damped(data, sampling_rate, lowpass_cutoff_frequency,
        order=4):
    """See Robertson, 2003. This code is transcribed from some MATLAB code that
    Amy Silder gave me. This implementation is slightly different from that
    appearing in Robertson, 2003. We only allow lowpass filtering.
    Parameters
    ----------
    data : array_like
        The signal to filter.
    sampling_rate : float
    lowpass_cutoff_frequency : float
        In Hertz (not normalized).
    order : int, optional
        Number of filter passes.
    Returns
    -------
    data : array_like
        Filtered data.
    """
    # 3 dB cutoff correction.
    Clp = (2.0 ** (1.0 / (2.0 * order)) - 1.0) ** (-0.5)

    # Corrected cutoff frequency.
    flp = Clp * lowpass_cutoff_frequency / sampling_rate

    # Warp cutoff frequency from analog to digital domain.
    wolp = np.tan(np.pi * flp)

    # Filter coefficients, K1 and K2.
    # lowpass: a0 = A0, a1 = A1, a2 = A2, b1 = B2, b2 = B2
    K1lp = 2.0 * wolp
    K2lp = wolp ** 2

    # Filter coefficients.
    a0lp = K2lp / (1.0 + K1lp + K2lp)
    a1lp = 2.0 * a0lp
    a2lp = a0lp
    b1lp = 2.0 * a0lp  * (1.0 / K2lp - 1.0)
    b2lp = 1.0 - (a0lp + a1lp + a2lp + b1lp)

    num_rows = len(data)
    temp_filtered = np.zeros(num_rows)
    # For order = 4, we go forward, backward, forward, backward.
    for n_pass in range(order):
        for i in range(2, num_rows):
            temp_filtered[i] = (a0lp * data[i] +
                    a1lp * data[i - 1] +
                    a2lp * data[i - 2] +
                    b1lp * temp_filtered[i - 1] +
                    b2lp * temp_filtered[i - 2])
        # Perform the filter backwards.
        data = np.flipud(temp_filtered)
        temp_filtered = np.zeros(num_rows)

    return data
