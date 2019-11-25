import matplotlib.pyplot as plt
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.trajectories import PiecewisePolynomial

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.multibody.parsing import Parser
from pydrake.systems.primitives import SignalLogger
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant


# Create a cubic polynomial that connects x_start and x_end.
# x_start and x_end should be list or np arrays.
def ConnectPointsWithCubicPolynomial(x_start, x_end, duration):
    t_knots = [0, duration / 2, duration]
    n = len(x_start)
    assert n == len(x_end)
    x_knots = np.zeros((3, n))
    x_knots[0] = x_start
    x_knots[2] = x_end
    x_knots[1] = (x_knots[0] + x_knots[2]) / 2
    return PiecewisePolynomial.Cubic(
        t_knots, x_knots.T, np.zeros(n), np.zeros(n))


def subsample_from_length_to_n(n, length):
    a = np.arange(0, length, int(round(length / n)))
    a = np.hstack((a, length - 1))
    return a


def PlotJointLog(iiwa_joint_log: SignalLogger, legend: str, y_label: str):
    '''
    Plots per-joint quantities from its signal logger system.
    '''
    fig = plt.figure(figsize=(8, 14), dpi=100)

    t = iiwa_joint_log.sample_times()
    num_plot_poins = 1000
    n = len(t)
    indices = subsample_from_length_to_n(num_plot_poins, n)

    for i, signal in enumerate(iiwa_joint_log.data()):
        ax = fig.add_subplot(711 + i)
        ax.plot(t[indices], signal[indices], label=legend + str(i + 1))
        ax.set_xlabel("t(s)")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log):
    '''
    Plots iiwa_position from signal logger systems.
    '''
    fig = plt.figure(figsize=(8, 14), dpi=100)
    t = iiwa_position_command_log.sample_times()
    num_plot_poins = 1000
    n = len(t)
    indices = subsample_from_length_to_n(num_plot_poins, n)

    for i in range(len(iiwa_position_command_log.data())):
        ax = fig.add_subplot(711 + i)
        q_commanded = iiwa_position_command_log.data()[i]
        q_measured = iiwa_position_measured_log.data()[i]
        ax.plot(t[indices], q_commanded[indices] / np.pi * 180,
                label='q_cmd@%d' % (i + 1))
        ax.plot(t[indices], q_measured[indices] / np.pi * 180,
                label='q@%d' % (i + 1))
        ax.set_xlabel("t(s)")
        ax.set_ylabel("degrees")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def GetPlanStartingTimes(kuka_plans):
    """
    :param kuka_plans: a list of Plans.
    :return: t_plan is a list of length (len(kuka_plans) + 1). t_plan[i] is
        the starting time of kuka_plans[i]; t_plan[-1] is the time at which
        the last plan ends.
    """
    num_plans = len(kuka_plans)
    t_plan = np.zeros(num_plans + 1)
    for i in range(0, num_plans):
        t_plan[i + 1] = \
            t_plan[i] + kuka_plans[i].get_duration() + 1.0
    return t_plan


def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def CalcL7PoseInWolrdFrame(iiwa_position_measured_log, num_plot_points=None, 
                           n_start=0, n_end=None):
    plant, _ = CreateIiwaControllerPlant()
    context = plant.CreateDefaultContext()
    l7_frame = plant.GetFrameByName('iiwa_link_7')

    t = iiwa_position_measured_log.sample_times()
    if n_end is None:
        n_end = len(t)

    t = iiwa_position_measured_log.sample_times()[n_start:n_end]
    q_log = iiwa_position_measured_log.data().T[n_start:n_end]
    
    if num_plot_points is not None:
        # sub sample points for plot
        n = len(t)
        indices = subsample_from_length_to_n(num_plot_poins, n)
        t = t[indices]
        q_log = q_log[indices]


    list_X_WL7 = list()
    for q in q_log:
        plant.SetPositions(context, q)
        X_WL7 = plant.CalcRelativeTransform(
                    context, frame_A=plant.world_frame(),
                    frame_B=l7_frame)
        list_X_WL7.append(X_WL7)

    return list_X_WL7, t


def CalcQPositionInWorldFrame(list_X_WL7, p_L7Q):
    p_WQ = np.zeros((len(list_X_WL7), 3))
    for i, X_WL7 in enumerate(list_X_WL7):
        p_WQ[i] = X_WL7.multiply(p_L7Q)

    return p_WQ


def PlotEeOrientationError(iiwa_position_measured_log, Q_WL7_ref, t_plan):
    """ Plots the absolute value of rotation angle between frame L7 and its reference.
    Q_WL7_ref is a quaternion of frame L7's reference orientation relative to world frame.
    t_plan is the starting time of every plan. They are plotted as vertical dashed black lines.  
    """
    station = ManipulationStation()
    station.SetupManipulationClassStation()
    station.Finalize()

    plant_iiwa = station.get_controller_plant()
    tree_iiwa = plant_iiwa.tree()
    context_iiwa = plant_iiwa.CreateDefaultContext()
    l7_frame = plant_iiwa.GetFrameByName('iiwa_link_7')

    t_sample = iiwa_position_measured_log.sample_times()
    n = len(t_sample)
    angle_error_abs = np.zeros(n - 1)
    for i in range(1, n):
        q_iiwa = iiwa_position_measured_log.data()[:, i]
        x_iiwa_mutable = \
            tree_iiwa.GetMutablePositionsAndVelocities(context_iiwa)
        x_iiwa_mutable[:7] = q_iiwa

        X_WL7 = tree_iiwa.CalcRelativeTransform(
            context_iiwa, frame_A=plant_iiwa.world_frame(), frame_B=l7_frame)

        Q_L7L7ref = X_WL7.quaternion().inverse().multiply(Q_WL7_ref)
        angle_error_abs[i - 1] = np.arccos(Q_L7L7ref.w()) * 2

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    ax.axhline(0, linestyle='--', color='r')
    for t in t_plan:
        ax.axvline(t, linestyle='--', color='k')
    ax.plot(t_sample[1:], angle_error_abs / np.pi * 180)
    ax.set_xlabel("t(s)")
    ax.set_ylabel("abs angle error, degrees")

    plt.tight_layout()
    plt.show()


def CreateIiwaControllerPlant():
    # creates plant that includes only the robot, used for controllers.
    robot_sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    parser = Parser(plant=plant_robot)
    parser.AddModelFromFile(robot_sdf_path)
    plant_robot.WeldFrames(
        A=plant_robot.world_frame(),
        B=plant_robot.GetFrameByName("iiwa_link_0"))
    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index())

    return plant_robot, link_frame_indices




