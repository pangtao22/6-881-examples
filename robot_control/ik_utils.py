import numpy as np

import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mathematicalprogram import SolutionResult
from pydrake.multibody import inverse_kinematics
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.trajectories import PiecewisePolynomial
from manip_station_sim.plan_utils import CreateIiwaControllerPlant


# Create MultibodyPlant for IK solves
plant, _ = CreateIiwaControllerPlant()
world_frame = plant.world_frame()
l7_frame = plant.GetFrameByName("iiwa_link_7")


def InverseKinPointwise(InterpolatePosition,
                        InterpolateOrientation,
                        p_L7Q,
                        duration,
                        num_knot_points,
                        q_initial_guess,
                        position_tolerance=0.005,
                        theta_bound=0.005 * np.pi):
    """
    Calculates a joint space trajectory for iiwa by repeatedly calling IK.
    The returned trajectory has (num_knot_points) knot points. To improve 
    the continuity of the trajectory, the IK from which q_knots[i] is
    solved is initialized with q_knots[i-1].

    Positions for point Q (p_EQ) and orientations for the end effector, generated
        respectively by InterpolatePosition and InterpolateOrientation,
        are added to the IKs as constraints.

    :param InterpolatePosition: A function with signature (tau) with
            0 <= tau <= 1.
        It returns p_WQ, a (3,) numpy array which describes the desired
        position of Q. For example, to get p_WQ for knot point i, use
            tau = i / (num_knot_points - 1).
    :param InterpolateOrientation: A function with the same signature as
                InterpolatePosition.
        It returns R_WL7, a RotationMatrix which describes the desired
        orientation of frame L7.
    :param num_knot_points: number of knot points in the trajectory.
    :param q_initial_guess: initial guess for the first IK.

    :param position_tolerance: tolerance for IK position constraints in meters.
    :param theta_bound: tolerance for IK orientation constraints in radians.
    :param is_printing: whether the solution results of IKs are printed.
    :return: qtraj: a 7-dimensional cubic polynomial that describes a
        trajectory for the iiwa arm.
    :return: q_knots: a (n, num_knot_points) numpy array (where n =
        plant.num_positions()) that stores solutions returned by all IKs. It
        can be used to initialize IKs for the next trajectory.
    """
    q_knots = np.zeros((num_knot_points, plant.num_positions()))

    ################### Your code here ###################
    for i in range(num_knot_points):
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()

        # Orientation constraint
        R_WL7_ref = InterpolateOrientation(i / (num_knot_points - 1))
        ik.AddOrientationConstraint(
            frameAbar=world_frame, R_AbarA=R_WL7_ref,
            frameBbar=l7_frame, R_BbarB=RotationMatrix.Identity(),
            theta_bound=theta_bound)

        # Position constraint
        p_WQ = InterpolatePosition(i / (num_knot_points - 1))
        ik.AddPositionConstraint(
            frameB=l7_frame, p_BQ=p_L7Q,
            frameA=world_frame,
            p_AQ_lower=p_WQ - position_tolerance,
            p_AQ_upper=p_WQ + position_tolerance)

        prog = ik.prog()
        # use the robot configuration at the previous knot point as
        # an initial guess.
        if i > 0:
            prog.SetInitialGuess(q_variables, q_knots[i-1])
        else:
            prog.SetInitialGuess(q_variables, q_initial_guess)
        result = mp.Solve(prog)

        # throw if no solution found.
        if result.get_solution_result() != SolutionResult.kSolutionFound:
            print(i, result.get_solution_result())
            raise RuntimeError

        q_knots[i] = result.GetSolution(q_variables)

    t_knots = np.linspace(0, duration, num_knot_points)
    qtraj = PiecewisePolynomial.Cubic(
        t_knots, q_knots.T, np.zeros(7), np.zeros(7))

    return qtraj, q_knots
    ########################################################
    raise NotImplementedError

def SolveOneShotIk(
        p_WQ_ref, 
        R_WL7_ref, 
        p_L7Q,
        q_initial_guess,
        position_tolerance=0.005,
        theta_bound=0.005):
    ik_scene = inverse_kinematics.InverseKinematics(plant)

    ik_scene.AddOrientationConstraint(
        frameAbar=world_frame, R_AbarA=R_WL7_ref,
        frameBbar=l7_frame, R_BbarB=RotationMatrix.Identity(),
        theta_bound=theta_bound)

    p_WQ_lower = p_WQ_ref - position_tolerance
    p_WQ_upper = p_WQ_ref + position_tolerance
    ik_scene.AddPositionConstraint(
        frameB=l7_frame, p_BQ=p_L7Q,
        frameA=world_frame,
        p_AQ_lower=p_WQ_lower, p_AQ_upper=p_WQ_upper)

    prog = ik_scene.prog()
    prog.SetInitialGuess(ik_scene.q(), q_initial_guess)
    result = mp.Solve(prog)
    if result.get_solution_result() != SolutionResult.kSolutionFound:
        print(result.get_solution_result())
        raise RuntimeError
    
    return result.GetSolution(ik_scene.q())

