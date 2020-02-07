#!/usr/bin/env python3
import argparse
import numpy as np
import opensim as osim

parser = argparse.ArgumentParser(
    description="Repeat a MocoTrajectory. The trajectory is assumed to be "
                "periodic (last time step matches first time step).")
parser.add_argument('input', help="A MocoTrajectory to repeat.")
parser.add_argument('count', type=int,
                    help="The resulting trajectory is COUNT times "
                         "longer than the original trajectory.")
parser.add_argument('output', help="The repeated MocoTrajectory.")
parser.add_argument('--add', type=str, nargs='+',
                    help="Paths of states whose value should accumulate with "
                         "each repeat.")

args = parser.parse_args()

if args.count < 2:
    raise RuntimeError("count must be 2 or greater.")

trajectory = osim.MocoTrajectory(args.input)

time0 = trajectory.getTimeMat()
duration0 = time0[-1] - time0[0]
# print('duration0:', duration0)
time = np.concatenate((np.tile(time0[:-1], args.count), [time0[-1]]))

N0 = trajectory.getNumTimes()

for count_index in range(1, args.count):
    start_row_index = (N0 - 1) * count_index
    end_row_index = (N0 - 1) * (count_index + 1)
    for row_index in range(start_row_index, end_row_index):
        time[row_index] += count_index * duration0
time[-1] += (args.count - 1) * duration0
# print('durationrepeated:', time[-1] - time[0])

states = trajectory.getStatesTrajectoryMat()
controls = trajectory.getControlsTrajectoryMat()
multipliers = trajectory.getMultipliersTrajectoryMat()
# derivatives = trajectory.getDerivativesTrajectoryMat()

N = (N0 - 1) * args.count + 1
trajectory.setNumTimes(N)
trajectory.setTime(time)

def repeat(names, trajmat0, setter, handle_add=False):
    traj_repeated = np.concatenate(
        (np.tile(trajmat0[:-1, :], (args.count, 1)),
         [trajmat0[-1, :]]
         ))
    if handle_add and args.add:
        for add_name in args.add:
            add_index = -1
            for iadd, name in enumerate(names):
                if name == add_name:
                    add_index = iadd
                    break
            if add_index < 0:
                raise RuntimeError(f"add name '{add_name}' not found.")
            diff = trajmat0[-1, add_index] - trajmat0[0, add_index]

            for count_index in range(1, args.count):
                start_row_index = (N0 - 1) * count_index
                end_row_index = (N0 - 1) * (count_index + 1)
                for row_index in range(start_row_index, end_row_index):
                    traj_repeated[row_index, add_index] += count_index * diff
            traj_repeated[-1, add_index] += (args.count - 1) * diff

    for i, name in enumerate(names):
        setter(name, traj_repeated[:, i])

repeat(trajectory.getStateNames(), states, trajectory.setState, True)
repeat(trajectory.getControlNames(), controls, trajectory.setControl)
repeat(trajectory.getMultiplierNames(), multipliers, trajectory.setMultiplier)
# repeat(trajectory.getDerivativeNames(), derivatives, trajectory.setDerivative)

trajectory.write(args.output)
