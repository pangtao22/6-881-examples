import numpy as np

from pydrake.common.eigen_geometry import AngleAxis
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.trajectories import PiecewisePolynomial
from pydrake.math import RigidTransform, RotationMatrix

from .plan_utils import (ConnectPointsWithCubicPolynomial,
                         CreateIiwaControllerPlant)

plan_type_strings = [
    "JointSpacePlan",
    "JointSpacePlanRelative",
    "JointSpacePlanGoToTarget",
    "IiwaTaskSpaceVelocityPlan",
    "IiwaTaskSpaceImpedancePlan"
]

PlanTypes = dict()
for plan_types_string in plan_type_strings:
    PlanTypes[plan_types_string] = plan_types_string


class PlanBase:
    def __init__(self,
                 type=None,
                 trajectory=None):
        self.type = type
        self.traj = trajectory
        self.traj_d = None
        self.duration = None
        self.start_time = None
        if trajectory is not None:
            self.traj_d = trajectory.derivative(1)
            self.duration = trajectory.end_time()

    def get_duration(self):
        return self.duration

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        pass

    def CalcTorqueCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                          control_period):
        return np.zeros(7)


class JointSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlan"],
                          trajectory=trajectory)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        return self.traj.value(t_plan).flatten()


class JointSpacePlanGoToTarget(PlanBase):
    """
    The robot goes to q_target from its configuration when this plan starts.
    """
    def __init__(self, duration, q_target):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanGoToTarget"],
                          trajectory=None)
        self.q_target = q_target
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.q_target, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_cmd)
        return self.traj.value(t_plan).flatten()


class JointSpacePlanRelative(PlanBase):
    """
    The robot goes from its configuration when this plan starts (q_current) by
    delta_q to reach the final configuration (q_current + delta_q).
    """
    def __init__(self, duration, delta_q):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanRelative"],
                          trajectory=None)
        self.delta_q = delta_q
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.delta_q + q_start, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_cmd)
        return self.traj.value(t_plan).flatten()


class IiwaTaskSpacePlan(PlanBase):
    def __init__(self,
                 calc_p_Q,
                 calc_R_WL7_ref,
                 duration: float,
                 p_L7Q: np.array):
        """
        :param calc_p_WQ a function of signature (tau) where 
            tau is the interpolation factor between 0 and 1. 
            It returns the reference xyz position of point Q in world frame, 
            RELATIVE TO ITS POSITION AT THE BEGINNING OF THE PLAN (p_WQ_0):
            p_WQ_ref(t) = calc_p_Q(t / duration) + p_WQ_0.
        :param calc_R_WL7_ref: a function of signature (tau) where
            tau is the interpolation factor between 0 and 1. 
            It returns the reference orientation of frame L7 relative 
            to world frame as a RotationMatrix.
        :param p_L7Q: the point in frame L7 that tracks xyz_traj. Its default
            value is the origin of L7.
        """
        PlanBase.__init__(self)

        self.p_WQ_0 = None
        self.q_iiwa_previous = np.zeros(7)

        # kinematics calculation
        self.plant, _ = CreateIiwaControllerPlant()
        self.context = self.plant.CreateDefaultContext()
        self.l7_frame = self.plant.GetFrameByName('iiwa_link_7')

        # Store EE rotation reference.
        self.calc_R_WL7_ref = calc_R_WL7_ref
        self.calc_p_Q = calc_p_Q
        self.p_L7Q = p_L7Q
        self.duration = duration

        # data members updated by CalcKinematics
        self.X_WL7 = None
        self.p_WQ = None
        self.R_WL7=None
        self.Jv_WL7q = None

    def UpdatePwq0(self, p_WQ_0):
        assert len(p_WQ_0) == 3
        self.p_WQ_0 = np.copy(p_WQ_0)

    def CalcKinematics(self, q_iiwa, v_iiwa):
        """
        @param q_iiwa: robot configuration.
        @param v_iiwa: robot velocity.
        Updates the following data members:
        - Jv_WL7q: geometric jacboain of point Q in frame L7.
        - p_WQ: position of point Q in world frame.
        - Q_WL7: orientation of frame L7 in world frame as a quaternion.
        - X_WL7: pose of frame L7 relative to world frame.
        """
        # update plant context
        self.plant.SetPositions(self.context, q_iiwa)
        self.plant.SetVelocities(self.context, v_iiwa)

        # Pose of frame L7 in world frame
        self.X_WL7 = self.plant.CalcRelativeTransform(
            self.context, frame_A=self.plant.world_frame(),
            frame_B=self.l7_frame)

        # Position of Q in world frame
        self.p_WQ = self.X_WL7.multiply(self.p_L7Q)

        # Orientation of L7 in world frame
        self.R_WL7 = self.X_WL7.rotation()

        # calculate Geometric jacobian (6 by 7 matrix) of point Q in frame L7.
        self.Jv_WL7q = self.plant.CalcJacobianSpatialVelocity(
            context=self.context,
            with_respect_to=JacobianWrtVariable.kQDot,
            frame_B=self.l7_frame,
            p_BP=self.p_L7Q,
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame())

    def CalcPositionError(self, t_plan):
        # must be called after calling CalcKinematics
        p_WQ_ref = self.p_WQ_0 + self.calc_p_Q(t_plan / self.duration)
        return p_WQ_ref - self.p_WQ

    def CalcOrientationError(self, t_plan):
        # must be called after calling CalcKinematics
        R_WL7_ref = self.calc_R_WL7_ref(t_plan / self.duration)
        R_L7L7r = self.R_WL7.inverse().multiply(R_WL7_ref)
        return R_L7L7r


class IiwaTaskSpaceVelocityPlan(IiwaTaskSpacePlan):
    """
    Refer to the base class constructor for more documentation.
    """
    def __init__(self,
                 calc_p_Q,
                 calc_R_WL7_ref,
                 duration: float,
                 p_L7Q: np.array):
        IiwaTaskSpacePlan.__init__(
            self,
            calc_p_Q,
            calc_R_WL7_ref,
            duration, p_L7Q)
        self.type = PlanTypes["IiwaTaskSpaceVelocityPlan"]

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        self.CalcKinematics(q_iiwa, v_iiwa)

        if self.p_WQ_0 is None:
            self.UpdatePwq0(self.p_WQ)

        if t_plan < self.duration:
            # position and orientation errors.
            err_xyz = self.CalcPositionError(t_plan)
            R_L7L7r = self.CalcOrientationError(t_plan)
            q_cmd = np.zeros(7)

            ############ Your code here #########################
            # first 3: angular velocity, last 3: translational velocity
            v_ee_desired = np.zeros(6)

            # Translation
            kp_translation = np.array([100., 100., 100])
            v_ee_desired[3:6] = kp_translation * err_xyz

            # Rotation
            kp_rotation = np.array([50., 50, 50])
            v_ee_desired[0:3] = self.R_WL7.multiply(
                kp_rotation * R_L7L7r.ToQuaternion().xyz())

            result = np.linalg.lstsq(self.Jv_WL7q, v_ee_desired, rcond=None)
            qdot_desired = np.clip(result[0], -1, 1)

            q_cmd = q_iiwa + qdot_desired * control_period
            #####################################################
            self.q_iiwa_previous[:] = q_iiwa
            return q_cmd
        else:
            return self.q_iiwa_previous


class IiwaTaskSpaceImpedancePlan(IiwaTaskSpacePlan):
    """
    Refer to the base class constructor for more documentation.
    """
    def __init__(self,
                 calc_p_Q,
                 calc_R_WL7_ref,
                 duration: float,
                 p_L7Q: np.array):
        IiwaTaskSpacePlan.__init__(
            self,
            calc_p_Q,
            calc_R_WL7_ref,
            duration, p_L7Q)
        self.type = PlanTypes["IiwaTaskSpaceImpedancePlan"]

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        if t_plan < self.duration:
            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa
        else:
            return self.q_iiwa_previous

    def CalcTorqueCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                          control_period):
        self.CalcKinematics(q_iiwa, v_iiwa)
        if self.p_WQ_0 is None:
            self.UpdatePwq0(self.p_WQ)

        if t_plan < self.duration:
            err_position = self.CalcPositionError(t_plan)
            R_L7L7r = self.CalcOrientationError(t_plan)
            tau_cmd = np.zeros(7)

            ############ Your code here #########################

            # first 3: angular velocity, last 3: translational velocity
            f_ee_desired = np.zeros(6)

            # translation
            kp_translation = np.array([100., 100., 100])#*15
            f_ee_desired[3:6] = kp_translation * err_position

            # rotation
            kp_rotation = np.array([50, 50, 50])
            f_ee_desired[0:3] = self.R_WL7.multiply(
                kp_rotation * R_L7L7r.ToQuaternion().xyz())

            tau_cmd = np.clip(self.Jv_WL7q.T.dot(f_ee_desired), -20, 20)
            #####################################################
            return tau_cmd
        else:
            return np.zeros(7)


