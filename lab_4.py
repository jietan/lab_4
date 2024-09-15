import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
np.set_printoptions(precision=3, suppress=True)


Kp = 3
Kd = 0.1

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics')
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.joint_subscription  # prevent unused variable warning

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_command_controller/commands',
            10
        )

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 50   # 50 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None

        touch_down_position = np.array([0.05, 0.0, -0.12])
        liftoff_position = np.array([-0.05, 0.0, -0.12])
        mid_swing_position = np.array([0.0, 0.0, -0.06])

        rf_ee_triangle_positions = np.array([
            touch_down_position,
            liftoff_position,
            mid_swing_position
        ]) + np.array([0.07500, -0.08350, 0])
        lf_ee_triangle_positions = np.array([
            mid_swing_position,
            touch_down_position,
            liftoff_position,
        ]) + np.array([0.07500, 0.08350, 0])
        rb_ee_triangle_positions = np.array([
            mid_swing_position,
            touch_down_position,
            liftoff_position,
        ]) + np.array([-0.07500, -0.07250, 0])
        lb_ee_triangle_positions = np.array([
            touch_down_position,
            liftoff_position,
            mid_swing_position
        ]) + np.array([-0.07500, 0.08350, 0])

        # (4, 3, 3)
        self.ee_triangle_positions = np.stack([rf_ee_triangle_positions, lf_ee_triangle_positions, rb_ee_triangle_positions, lb_ee_triangle_positions])

        self.current_target = 0
        self.t = 0

    def listener_callback(self, msg):
        joints_of_interest = ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3', 'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3', 'leg_back_r_1', 'leg_back_r_2', 'leg_back_r_3', 'leg_back_l_1', 'leg_back_l_2', 'leg_back_l_3']
        self.joint_positions = np.array([msg.position[msg.name.index(joint)] for joint in joints_of_interest])
        self.joint_velocities = np.array([msg.velocity[msg.name.index(joint)] for joint in joints_of_interest])

    def forward_kinematics(self, theta):
        def rotation_x(angle):
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), 0],
                [0, 0, 0, 1]
            ])

        def rotation_y(angle):
            return np.array([
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])

        def rotation_z(angle):
            return np.array([
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        def translation(x, y, z):
            return np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ])

        T_RF_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RF_3_ee = translation(0.06231, -0.06216, 0.01800)

        T_LF_0_1 = translation(0.07500, 0.08350, 0) @ rotation_x(-1.57080) @ rotation_z(theta[3])
        T_LF_1_2 = rotation_y(1.57080) @ rotation_z(theta[4])
        T_LF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[5])
        T_LF_3_ee = translation(0.06231, 0.06216, 0.01800)

        T_RB_0_1 = translation(-0.07500, -0.07250, 0) @ rotation_x(1.57080) @ rotation_z(theta[6])
        T_RB_1_2 = rotation_y(-1.57080) @ rotation_z(theta[7])
        T_RB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[8])
        T_RB_3_ee = translation(0.06231, -0.06216, 0.01800)

        T_LB_0_1 = translation(-0.07500, 0.08350, 0) @ rotation_x(-1.57080) @ rotation_z(theta[9])
        T_LB_1_2 = rotation_y(1.57080) @ rotation_z(theta[10])
        T_LB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[11])
        T_LB_3_ee = translation(0.06231, 0.06216, 0.01800)

        T_RF_0_ee = T_RF_0_1 @ T_RF_1_2 @ T_RF_2_3 @ T_RF_3_ee
        T_LF_0_ee = T_LF_0_1 @ T_LF_1_2 @ T_LF_2_3 @ T_LF_3_ee
        T_RB_0_ee = T_RB_0_1 @ T_RB_1_2 @ T_RB_2_3 @ T_RB_3_ee
        T_LB_0_ee = T_LB_0_1 @ T_LB_1_2 @ T_LB_2_3 @ T_LB_3_ee

        return np.concatenate([T_RF_0_ee[:3, 3], T_LF_0_ee[:3, 3], T_RB_0_ee[:3, 3], T_LB_0_ee[:3, 3]])

    def inverse_kinematics(self, target_ee, initial_guess=np.zeros(12)):
        def cost_function(theta):
            current_position = self.forward_kinematics(theta)
            l1 = np.abs(current_position - target_ee)
            cost = np.sum(l1**2)
            return cost, l1

        def gradient(theta, epsilon=1e-3):
            grad = np.zeros(12)
            for i in range(12):
                theta_plus = theta.copy()
                theta_plus[i] += epsilon
                theta_minus = theta.copy()
                theta_minus[i] -= epsilon
                grad[i] = (cost_function(theta_plus)[0] - cost_function(theta_minus)[0]) / (2 * epsilon)
            return grad

        theta = np.array(initial_guess)
        learning_rate = 10
        max_iterations = 50
        tolerance = 1e-3

        cost_l = []
        for _ in range(max_iterations):
            grad = gradient(theta)
            theta -= learning_rate * grad
            cost, l1 = cost_function(theta)
            cost_l.append(cost)
            if l1.mean() < tolerance:
                break

        return theta

    def interpolate_triangle(self, t):
        t = t % 1  # Cycle through 3 seconds
        t *= 3
        if t < 1:
            return (t * self.ee_triangle_positions[:, 0] + (1-t) * self.ee_triangle_positions[:, 1]).reshape(-1)
        elif t < 2:
            t -= 1
            return (t * self.ee_triangle_positions[:, 1] + (1-t) * self.ee_triangle_positions[:, 2]).reshape(-1)
        else:
            t -= 2
            return (t * self.ee_triangle_positions[:, 2] + (1-t) * self.ee_triangle_positions[:, 0]).reshape(-1)

    def ik_timer_callback(self):
        if self.joint_positions is not None:
            target_ee = self.interpolate_triangle(self.t)
            self.target_joint_positions = self.inverse_kinematics(target_ee, self.joint_positions)
            current_ee = self.forward_kinematics(self.joint_positions)

            self.t += self.ik_timer_period
            self.get_logger().info(f'Target EE: {target_ee}, Current EE: {current_ee}, Target Angles: {self.target_joint_positions}, Target Angles to EE: {self.forward_kinematics(self.target_joint_positions)}, Current Angles: {self.joint_positions}')

    def pd_timer_callback(self):
        if self.joint_positions is not None and self.joint_velocities is not None and self.target_joint_positions is not None:
            torques = Kp * (self.target_joint_positions - self.joint_positions) - Kd * self.joint_velocities

            command_msg = Float64MultiArray()
            command_msg.data = torques.tolist()
            self.command_publisher.publish(command_msg)

def main():
    rclpy.init()
    inverse_kinematics = InverseKinematics()
    
    try:
        rclpy.spin(inverse_kinematics)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Send zero torques
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0, 0.0, 0.0]
        inverse_kinematics.command_publisher.publish(zero_torques)
        
        inverse_kinematics.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
