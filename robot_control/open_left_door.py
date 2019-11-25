import numpy as np

import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mathematicalprogram import SolutionResult
from pydrake.multibody import inverse_kinematics
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.trajectories import PiecewisePolynomial

from manip_station_sim.plan_utils import (CreateIiwaControllerPlant,
                                          ConnectPointsWithCubicPolynomial)
from manip_station_sim.robot_plans import (JointSpacePlan, JointSpacePlanGoToTarget,
                                           JointSpacePlanRelative, IiwaTaskSpacePlan)

# open left door related constants  ----------------
# L: frame of the cupboard left door, whose origin is at the center of the door
# body.
# Position of the origin of L in world frame.
p_WQ_home = np.array([0.5, 0, 0.41])

# L: frame the cupboard left door, whose origin is at the center of the door body.
p_WL = np.array([0.7477, 0.1445, 0.4148]) #+ [-0.1, 0, 0]
# center of the left hinge of the door in frame L and W
p_LC_left_hinge = np.array([0.008, 0.1395, 0])
p_WC_left_hinge = p_WL + p_LC_left_hinge

# center of handle in frame L and W
p_LC_handle = np.array([-0.033, -0.1245, 0])
p_WC_handle = p_WL + p_LC_handle

# distance between the hinge center and the handle center
p_handle_2_hinge = p_LC_handle - p_LC_left_hinge
r_handle = np.linalg.norm(p_handle_2_hinge)

# angle between the world y axis and the line connecting the hinge cneter to the
# handle center when the left door is fully closed (left_hinge_angle = 0).
theta0_hinge = np.arctan2(np.abs(p_handle_2_hinge[0]),
                          np.abs(p_handle_2_hinge[1]))

# position of point Q in L7 frame.  Point Q is fixed w.r.t frame L7.
# When the robot is upright (all joint angles = 0), axes of L7 are aligned with axes of world.
# The origin of end effector frame (plant.GetFrameByName('body')) is located at [0, 0, 0.114] in frame L7.
p_L7Q = np.array([0., 0., 0.1]) + np.array([0, 0, 0.114])

# orientation of end effector aligned frame
R_WL7_ref = RollPitchYaw(0, np.pi / 180 * 135, 0).ToRotationMatrix()

# initial joint angles of the robot for simulation.
q0 = np.array([0., 0.6, 0., -1.75, 0., 1., 0.])
q_home = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0])

# motion interpolation functions ----------------

angle_end = np.pi / 180 * 50
# pull handle along an arc
def InterpolateArc(tau):
    angle_start = theta0_hinge
    theta = angle_start + (angle_end - angle_start) * tau
    xyz_handle = -r_handle * np.array([np.sin(theta), np.cos(theta), 0])
    return p_WC_left_hinge + xyz_handle


def InterpolateYawAngle(tau): 
    yaw_angle = 0 + (angle_end - theta0_hinge) * tau
    return RollPitchYaw(0, np.pi/4*3, -yaw_angle).ToRotationMatrix()


def calc_p_Q(tau):
    ############## Your code here ##############
    radius = r_handle
    angle_start = theta0_hinge
    theta = angle_start + (angle_end - angle_start) * tau
    xyz_handle = -radius * np.array([np.sin(theta), np.cos(theta), 0])
    return p_WC_left_hinge + xyz_handle - p_WC_handle
    #############################################
    raise NotImplementedError


def calc_R_WL7_ref(tau):
    ############## Your code here ##############
    yaw_angle = 0 + (angle_end - theta0_hinge) * tau
    return RollPitchYaw(0, np.pi/4*3, -yaw_angle).ToRotationMatrix()
    #############################################
    raise NotImplementedError


# utility functions --------------------------------------

def CalcHandleArcPoints():
    theta = np.linspace(0, np.pi/2, 100)
    points_on_handle_arc = np.zeros((100, 3))
    points_on_handle_arc[:, 0] = -np.sin(theta) * r_handle
    points_on_handle_arc[:, 1] = -np.cos(theta) * r_handle
    points_on_handle_arc += p_WC_left_hinge

    return points_on_handle_arc


def CalcRadialTrackingError(p_WQ):
    x = p_WQ[:, 0] - p_WC_left_hinge[0]
    y = p_WQ[:, 1] - p_WC_left_hinge[1]
    r = np.sqrt(x**2 + y**2)
    r_error = r - r_handle
    return r_error


def CalcOrientationTrackingError(list_X_WL7, t, duration):
    angle_error_list = []
    for i, X_WL7 in enumerate(list_X_WL7):
        tau = (t[i] - t[0]) / duration
        R_WL7_ref = InterpolateYawAngle(tau)
        R_WL7 = X_WL7.rotation()
        Q_error = R_WL7.inverse().multiply(R_WL7_ref).ToQuaternion()
        angle_error_list.append(2 * np.arccos(Q_error.w()))

    return np.array(angle_error_list)




