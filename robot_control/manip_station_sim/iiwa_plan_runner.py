from pydrake.systems.framework import (BasicVector, LeafSystem, PortDataType,
    AbstractValue, LeafSystem, PublishEvent, TriggerType)
from .robot_plans import *


class IiwaPlanRunner(LeafSystem):
    """ A drake system that sends commands to the robot and gripper by evaluating the currently
    active Plan.

    The plan runner is constructed with a list of Plans (kuka_plans) and a list of gripper
    setpoints (gripper_setpoint_list).

    In its constructor, it adds two additional plans to kuka_plans for safety reasons:
    - The first plan holds the robot's current position for 3 seconds.
    - The second plan moves the robot from its current position, to the position at the beginning of
        the first plan in the Plans list.

    Plans in the modified kuka_plans are then activated in sequence.
    Each plan is active for plan.duration seconds.

    By default, the plan runner sends position and torque commands to iiwa at 200Hz,
    and gripper setpoint commands to the schunk WSG50 at a lower rate.
    At every update event, the commands are generated by evaluating the currently active plan.

    The current implementation requires either
    - kuka_plans be an empty list, or
    - kuka_plans[0].traj be a valid 7-dimensional PiecewisePolynomial.
    """

    def __init__(self, iiwa_plans, gripper_setpoint_list,
                 control_period=0.005, print_period=1.0,
                 gripper_setpoint_initial=0.025):
        LeafSystem.__init__(self)
        assert len(iiwa_plans) == len(gripper_setpoint_list)
        self.set_name("IiwaPlanRunner")

        # Add a zero order hold to hold the current position of the robot
        iiwa_plans.insert(0, JointSpacePlanRelative(
            duration=1.0, delta_q=np.zeros(7)))
        gripper_setpoint_list.insert(0, gripper_setpoint_initial)

        if len(iiwa_plans) > 1:
            # Insert to the beginning of plan_list a plan that moves the robot from its
            # current position to plan_list[0].traj.value(0)
            iiwa_plans.insert(1, JointSpacePlanGoToTarget(
                duration=6.0, q_target=iiwa_plans[1].traj.value(0).ravel()))
            gripper_setpoint_list.insert(0, gripper_setpoint_initial)

        self.gripper_setpoint_list = gripper_setpoint_list
        self.kuka_plans_list = iiwa_plans

        self.current_plan_start_time = 0.
        self.current_plan = None
        self.current_gripper_setpoint = None
        self.current_plan_idx = 0

        # Stuff for iiwa control
        self.nq = 7
        self.print_period = print_period
        self.last_print_time = -print_period
        self.control_period = control_period

        # iiwa position input port
        self.iiwa_position_input_port = self.DeclareInputPort(
            "iiwa_position", PortDataType.kVectorValued, self.nq)

        # iiwa position commanded input port
        self.iiwa_position_command_input_port = self.DeclareInputPort(
            "iiwa_position_cmd", PortDataType.kVectorValued, self.nq)

        # iiwa velocity input port
        self.iiwa_velocity_input_port = self.DeclareInputPort(
            "iiwa_velocity", PortDataType.kVectorValued, self.nq)

        # iiwa external torque input port
        self.iiwa_external_torque_input_port = self.DeclareInputPort(
            "iiwa_torque_external", PortDataType.kVectorValued, self.nq)

        # position and torque command output port
        self.iiwa_position_command_output_port = self.DeclareVectorOutputPort(
            "iiwa_position_command", BasicVector(self.nq),
            self.CalcIiwaPositionCommand, {self.all_state_ticket()})
        self.iiwa_torque_command_output_port = self.DeclareVectorOutputPort(
            "iiwa_torque_command", BasicVector(self.nq),
            self.CalcIiwaTorqueCommand, {self.all_state_ticket()})

        # gripper setpoint and torque limit
        self.hand_setpoint_output_port = self.DeclareVectorOutputPort(
                "gripper_setpoint", BasicVector(1),
                self.CalcGripperSetpointOutput, {self.all_state_ticket()})

        self.gripper_force_limit_output_port = self.DeclareVectorOutputPort(
            "force_limit", BasicVector(1), self.CalcForceLimitOutput,
            {self.all_state_ticket()})

        # Declare command publishing rate
        # state[0:7]: position command
        # state[7:14]: torque command
        # state[14]: gripper_setpoint
        self.DeclareDiscreteState(self.nq * 2 + 1)
        self.DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)

    def GetCurrentPlan(self, context):
        t = context.get_time()

        if self.current_plan is None:
            # This is true only after the constructor is called and at the first control
            # tick after the simulator starts.
            self.current_plan = self.kuka_plans_list.pop(0)
            self.current_gripper_setpoint = self.gripper_setpoint_list.pop(0)
            self.current_plan_start_time = 0.
            return

        adjusted_duration = self.current_plan.duration + 1.0
        if t - self.current_plan_start_time >= adjusted_duration:
            if len(self.kuka_plans_list) > 0:
                self.current_plan = self.kuka_plans_list.pop(0)
                self.current_gripper_setpoint = self.gripper_setpoint_list.pop(0)
            else:
                # There are no more available plans. Hold current position.
                self.current_plan = JointSpacePlanRelative(
                    duration=3600., delta_q=np.zeros(7))
                if self.print_period < np.inf:
                    print('No more plans to run, holding current position...\n')

            self.current_plan_start_time = t
            self.current_plan_idx += 1

            if self.print_period < np.inf:
                print('Running plan %d' % self.current_plan_idx + " (type: " +
                      self.current_plan.type + \
                      "), starting at %f for a duration of %f seconds." % \
                      (t, self.current_plan.duration) + "\n")

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem.DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        self.GetCurrentPlan(context)

        t = context.get_time()
        q_iiwa = self.iiwa_position_input_port.Eval(context)
        v_iiwa = self.iiwa_velocity_input_port.Eval(context)
        tau_iiwa = self.iiwa_external_torque_input_port.Eval(context)
        q_iiwa_cmd = self.iiwa_position_command_input_port.Eval(context)
        t_plan = t - self.current_plan_start_time

        new_control_output = discrete_state.get_mutable_vector().get_mutable_value()

        new_control_output[0:self.nq] = \
            self.current_plan.CalcPositionCommand(q_iiwa, q_iiwa_cmd, v_iiwa, tau_iiwa,
                                                  t_plan, self.control_period)
        new_control_output[self.nq:2 * self.nq] = \
            self.current_plan.CalcTorqueCommand(q_iiwa, q_iiwa_cmd, v_iiwa, tau_iiwa,
                                                t_plan, self.control_period)
        new_control_output[14] = self.current_gripper_setpoint

        # print current simulation time
        if self.print_period < np.inf and \
                t - self.last_print_time >= self.print_period:
            print("t: ", t)
            self.last_print_time = t

    def CalcIiwaPositionCommand(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[0:self.nq]

    def CalcIiwaTorqueCommand(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[self.nq:2 * self.nq]

    def CalcGripperSetpointOutput(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[14]

    def CalcForceLimitOutput(self, context, output):
        output.SetAtIndex(0, 15.0)
