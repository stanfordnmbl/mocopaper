import re
import sys
import os

import numpy as np
from scipy.signal import butter, filtfilt

import pylab as pl
pl.matplotlib.use('TkAgg')

from utilities import ANCFile, \
    remove_fields_from_structured_ndarray, ndarray2storage, \
    filter_critically_damped
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

n_force_plates = 3

sides = ['r', 'l']

# TODO centers of pressure are computed differently than the GRF.
# TODO how many times do we need to zero out the data?
# TODO critically damped order? change to 4, or don't use?
# TODO new zeroing scheme for swing. See left Fz.
# TODO Fx has a forward component just at foot strike briefly. Weird...

# TODO new force zeroing scheme
# TODO new cop calculation after filtering
# TODO new cop interpolation

# This comes from Amy's settings in the Cortex software:
# 1000 means that the max voltage corresponds to 1000 N.
force_plate_calibration = [
        [1000, 1000, 2000, 600, 400, 300], # force plate 1.
        [1000, 1000, 2000, 900, 600, 450], # force plate 2.
        [1000, 1000, 2000, 600, 400, 300]  # force plate 3.
        ]
# Positions of force plate frame origins (center) from lab origin (meters).
force_plate_location = [
        [0.300, 0.200, -0.006],
        [0.901, 0.200, -0.006],
        [1.501, 0.200, -0.006]
        ]
# Orientation of force plate frames in the motion capture frame (x forward, y
# left, z up).
force_plate_rotation = [
        np.matrix([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]]),
        np.matrix([[-1.0, 0.0, 0.0],
                   [0.0,  1.0, 0.0],
                   [0.0, 0.0, -1.0]]),
        np.matrix([[0.0, -1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])
        ]
# Orientation of OpenSim ground frame (x forward, y up, z right) in motion
# capture frame.
opensim_rotation = np.matrix('1.0, 0.0, 0.0; 0.0, 0.0, -1.0; 0.0, 1.0, 0.0')

def transform_and_filter_ground_reaction(anc_fpath, mot_fpath, mot_name,
        order_of_foot_contact,
        mode='default',
        threshold=5,
        threshold2=5,
        butterworth_order=4,
        butterworth_cutoff_frequency=50,
        critically_damped_order=2,
        critically_damped_cutoff_frequency=6,
        gaussian_smoothing_sigma=5, 
        ):
    """Converts analog force plate data from the Stanford High Performance
    Laboratory (HPL)'s over-ground force plates into OpenSim .mot files. This
    includes filtering the force plate measurements, and computing center of
    pressure. This method is based on Amy Silder's write_GRF_MotFile.m.

    mode: options for number of times to cutoff and threshold/cutoff the data.
    """
    anc = ANCFile(anc_fpath)
    raw = anc.data

    out_path = os.path.split(mot_fpath)[0]

    pl.figure()

    n_times = anc['time'].shape[0]
    n_mot_times = n_times - critically_damped_order

    # Remove unnecessary columns (e.g., EMG signals).
    # -----------------------------------------------
    # Which columns do we want to remove? (e.g, EMG signals).
    fields_to_remove = list()
    for colname in anc.names:
        if (colname != 'time' and
                re.match('(F|M)[1-3](X|Y|Z)', colname) == None):
            fields_to_remove.append(colname)
    del colname

    # Remove the unwanted columns.
    data = remove_fields_from_structured_ndarray(raw, fields_to_remove).copy()


    # Unit conversion.
    # ----------------
    # Convert analog signal into volts.
    for colname in data.dtype.names:
        if colname != 'time':
            # Ranges are in millivolts.
            volts_per_bit = anc.ranges[colname] * 2 / 65536.0 * 0.001
            data[colname] *= volts_per_bit

    # Convert volts into forces (N) and moments (N-m).
    for i in range(n_force_plates):
        volts_to_forces = force_plate_calibration[i]
        fp = i + 1
        data['F%iX' % fp] *= volts_to_forces[0]
        data['F%iY' % fp] *= volts_to_forces[1]
        data['F%iZ' % fp] *= volts_to_forces[2]
        data['M%iX' % fp] *= volts_to_forces[3]
        data['M%iY' % fp] *= volts_to_forces[4]
        data['M%iZ' % fp] *= volts_to_forces[5]

    """
    # Filter forces and moments.
    for colname in data.dtype.names:
        if colname != 'time':
            # Create filter.
            nyquist_frequency = 0.5 * anc.rates[colname]
            normalized_butterworth_cutoff_frequency = \
                    butterworth_cutoff_frequency / nyquist_frequency
            lowpass_b, lowpass_a = butter(butterworth_order,
                    normalized_butterworth_cutoff_frequency)

            # Perform filter.
            data[colname] = filtfilt(lowpass_b, lowpass_a, data[colname])
    """

    # Initial filter.
    # ---------------
    if mode == 'default':
        # Critically damped filter to get rid of overshoot.
        # TODO may change array size.
        for colname in data.dtype.names:
            if colname != 'time':
                data[colname] = filter_critically_damped(
                        data[colname], anc.precise_rate,
                        critically_damped_cutoff_frequency,
                        order=critically_damped_order)

    def create_vecs(colnames):
        return np.array([list(item) for item in data[colnames]])
    def rotate_vecs(vecs, rotation_matrix):
        return vecs * rotation_matrix
    def repack_data(data, vecs, colnames):
        for icol, colname in enumerate(colnames):
            data[colname] = vecs[:, icol].reshape(-1)

    # Coordinate transformation.
    # --------------------------
    for fp in range(n_force_plates):
        force_names = ['F%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]
        moment_names = ['M%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]

        # Unpack 3 columns as rows of vectors.
        force_vecs = create_vecs(force_names)
        moment_vecs = create_vecs(moment_names)

        # Rotate.
        force_vecs = rotate_vecs(force_vecs, force_plate_rotation[fp])
        moment_vecs = rotate_vecs(moment_vecs, force_plate_rotation[fp])

        # Compute moments.
        # M_new = r x F + M_orig.
        r = np.tile(force_plate_location[fp], [n_times, 1])
        moment_vecs = np.cross(r, force_vecs) + moment_vecs

        # Pack back into the data.
        repack_data(data, force_vecs, force_names)
        repack_data(data, moment_vecs, moment_names)

    # Combine force plates to generate left foot, right foot data.
    # ------------------------------------------------------------
    forces = {side: np.zeros((n_times, 3)) for side in sides}
    moments = {side: np.zeros((n_times, 3)) for side in sides}

    assert (type(order_of_foot_contact) == str and
            len(order_of_foot_contact) == 3)

    for fp, foot_contact in enumerate(order_of_foot_contact):
        force_names = ['F%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]
        moment_names = ['M%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]

        forces[foot_contact] += create_vecs(force_names)
        moments[foot_contact] += create_vecs(moment_names)

    if mode == 'multicutoff':

        # Plot raw GRF (before cutting off or filtering).
        for side in sides:
            pl.plot(data['time'], -forces[side][:, 2],
                    label='F_y, %s, raw' % side, lw=0.5)
            pl.plot(data['time'], -forces[side][:, 0],
                    label='F_x, %s, raw' % side, lw=0.5)

        # Before filtering, cutoff the GRF using the first coarser cutoff.
        # Use Gaussian filter after cutoff to smooth force transitions at ground
        # contact and liftoff.
        for side in sides:
            filt = (forces[side][:, 2] > (-threshold))
            for item in [forces, moments]:
                item[side][filt, :] = 0
                for i in np.arange(item[side].shape[1]):
                    item[side][:,i] = gaussian_filter1d(item[side][:,i], 
                        gaussian_smoothing_sigma)

        # Critically damped filter (prevents overshoot).
        # TODO may change array size.
        for item in [forces, moments]:
            for side in sides:
                for direc in range(3):
                    item[side][:, direc] = filter_critically_damped(
                            item[side][:, direc], anc.precise_rate,
                            critically_damped_cutoff_frequency,
                            order=critically_damped_order)

    # Compute center of pressure.
    # ---------------------------
    # COPx = -My / Fz
    # COPy = Mx / Fz
    centers_of_pressure = {side: np.zeros((n_times, 3)) for side in sides}
    for side in sides:
        # Only compute when foot is on ground.
        # Time indices corresponding to foot on ground.
        filt = forces[side][:, 2] != 0
        Mx = moments[side][filt, 0]
        My = moments[side][filt, 1]
        Mz = moments[side][filt, 2]
        Fx = forces[side][filt, 0]
        Fy = forces[side][filt, 1]
        Fz = forces[side][filt, 2]
        COPx = - My / Fz
        COPy = Mx / Fz
        centers_of_pressure[side][filt, 0] = COPx
        centers_of_pressure[side][filt, 1] = COPy
        # Calculate Mz at the center of pressure.
        # Mz_new = Mz_old - COPx * Fy + COPy * Fx
        # Must have zero when foot is not on ground.
        Mz_new = np.zeros(n_times)
        Mz_new[filt] = Mz - COPx * Fy + COPy * Fx
        moments[side][:, 2] = Mz_new
        # Set Mx and My to 0 since they act at the center of pressure.
        moments[side][:, 0:2] = 0
        # TODO does this set all 3 columns to 0, or just the first 2?

    # Transform from motion capture frame to OpenSim ground frame.
    # ------------------------------------------------------------
    for side in sides:
        # Negate so that these are forces applied to the person.
        for item in [forces, moments]:
            item[side] = -item[side] * opensim_rotation
        centers_of_pressure[side] = \
                centers_of_pressure[side] * opensim_rotation

    # Spline interpolation of COP locations
    # -------------------------------------
    indices = np.arange(n_times)
    for side in sides:
        # Find the intervals where the foot is in swing.
        Fy = forces[side][:, 1]
        copx = centers_of_pressure[side][:, 0].A1
        copz = centers_of_pressure[side][:, 2].A1
        swing_filter = (Fy.A1 == 0)
        stance_filter = np.logical_not(swing_filter)
        for cop in [copx, copz]:
            cop[swing_filter] = np.nan
            spline = interpolate.InterpolatedUnivariateSpline(
                    indices[stance_filter], cop[stance_filter])
            cop[swing_filter] = spline(indices[swing_filter])
        centers_of_pressure[side][:, 0] = copx.reshape((copx.shape[0], -1))
        centers_of_pressure[side][:, 2] = copz.reshape((copz.shape[0], -1))


    # for item in [forces, moments]:
    #         for side in sides:
    #             for direc in range(3):
    #                 item[side][:, direc] = filter_critically_damped(
    #                         item[side][:, direc], anc.precise_rate,
    #                         critically_damped_cutoff_frequency,
    #                         order=critically_damped_order)


    # Finalize plot.
    # --------------
    # Plot processed GRF (before cutting off or filtering).
    for side in sides:
        pl.plot(data['time'], forces[side][:, 1], label='F_y, %s, processed' %
                side, lw=0.5)
        pl.plot(data['time'], forces[side][:, 0], label='F_x, %s, processed' %
                side, lw=0.5)
    pl.legend(frameon=False, loc='best')
    pl.savefig(os.path.join(out_path,
        'ground_reaction_before_and_after_processing.svg'))


    # Create structured array for mot file. Fill in the mot data.
    # -----------------------------------------------------------
    dtype_names = ['time']

    data_dict = dict()
    for side in sides:
        # Force.
        for idirec, direc in enumerate(['x', 'y', 'z']):
            colname = 'ground_force_%s_v%s' % (side, direc)
            dtype_names.append(colname)
            data_dict[colname] = forces[side][0:n_mot_times, idirec].reshape(-1)

        # Center of pressure.
        for idirec, direc in enumerate(['x', 'y', 'z']):
            colname = 'ground_force_%s_p%s' % (side, direc)
            dtype_names.append(colname)
            data_dict[colname] = \
                    centers_of_pressure[side][0:n_mot_times, idirec].reshape(-1)

        # Moment.
        for idirec, direc in enumerate(['x', 'y', 'z']):
            colname = 'ground_torque_%s_%s' % (side, direc)
            dtype_names.append(colname)
            data_dict[colname] = \
                    moments[side][0:n_mot_times, idirec].reshape(-1)

    mot_data = np.empty(n_mot_times, dtype={'names': dtype_names,
        'formats': len(dtype_names) * ['f8']})
    # TODO discrepancy with Amy.
    mot_data['time'] = data['time'][0:n_mot_times] #[[0] + range(n_mot_times-1)]
    for k, v in data_dict.items():
        mot_data[k] = v

    ndarray2storage(mot_data, mot_fpath, name=mot_name)


if __name__ == '__main__':
    transform_and_filter_ground_reaction(*sys.argv[1:])
