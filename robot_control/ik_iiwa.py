import numpy as np

from pydrake.multibody.plant import MultibodyPlant
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mathematicalprogram import SolutionResult
from pydrake.multibody import inverse_kinematics
from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform
from manip_station_sim.plan_utils import CreateIiwaControllerPlant

# Create MultibodyPlant for IK solves
plant, _ = CreateIiwaControllerPlant()
world_frame = plant.world_frame()
l7_frame = plant.GetFrameByName("iiwa_link_7")

# initial joint angles of the robot for simulation.
q0 = np.array([0., 0.6, 0., -1.75, 0., 1., 0.])
q_home = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0])

# object pose
X_WO = RigidTransform()
X_WO.set_translation([0.55, -0.1, -0.01])

p_L7Q = np.array([0., 0., 0.214])


def SolveOneShotIk(
        p_WQ_ref, 
        R_WL7_ref, 
        p_L7Q,
        q_initial_guess,
        position_tolerance=0.005,
        theta_bound=0.005):
    ik = inverse_kinematics.InverseKinematics(plant)

    ik.AddOrientationConstraint(
        frameAbar=world_frame, R_AbarA=R_WL7_ref,
        frameBbar=l7_frame, R_BbarB=RotationMatrix.Identity(),
        theta_bound=theta_bound)

    p_WQ_lower = p_WQ_ref - position_tolerance
    p_WQ_upper = p_WQ_ref + position_tolerance
    ik.AddPositionConstraint(
        frameB=l7_frame, p_BQ=p_L7Q,
        frameA=world_frame,
        p_AQ_lower=p_WQ_lower, p_AQ_upper=p_WQ_upper)

    prog = ik.prog()
    prog.SetInitialGuess(ik.q(), q_initial_guess)
    result = mp.Solve(prog)
    if result.get_solution_result() != SolutionResult.kSolutionFound:
        print(result.get_solution_result())
        raise RuntimeError
    
    return result.GetSolution(ik.q())

