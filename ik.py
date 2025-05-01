import numpy as np

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

def fr_leg_fk(theta):
    T_RF_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    T_RF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_RF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
    T_RF_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_RF_0_ee = T_RF_0_1 @ T_RF_1_2 @ T_RF_2_3 @ T_RF_3_ee
    return T_RF_0_ee[:3, 3]

def fl_leg_fk(theta):
    T_LF_0_1 = translation(0.07500, 0.08350, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    T_LF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_LF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    T_LF_3_ee = translation(0.06231, -0.06216, -0.01800)
    T_LF_0_ee = T_LF_0_1 @ T_LF_1_2 @ T_LF_2_3 @ T_LF_3_ee
    return T_LF_0_ee[:3, 3]

def br_leg_fk(theta):
    T_RB_0_1 = translation(-0.07500, -0.07250, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    T_RB_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_RB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
    T_RB_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_RB_0_ee = T_RB_0_1 @ T_RB_1_2 @ T_RB_2_3 @ T_RB_3_ee
    return T_RB_0_ee[:3, 3]

def lb_leg_fk(theta):
    T_LB_0_1 = translation(-0.07500, 0.07250, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    T_LB_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_LB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    T_LB_3_ee = translation(0.06231, -0.06216, -0.01800)
    T_LB_0_ee = T_LB_0_1 @ T_LB_1_2 @ T_LB_2_3 @ T_LB_3_ee
    return T_LB_0_ee[:3, 3]

def forward_kinematics(theta):
    fk_functions = [fr_leg_fk, fl_leg_fk, br_leg_fk, lb_leg_fk]
    return np.concatenate([fk_functions[i](theta[3*i: 3*i+3]) for i in range(4)])

def inverse_kinematics_single_leg(target_ee, leg_index, initial_guess=[0, 0, 0]):
    fk_functions = [fr_leg_fk, fl_leg_fk, br_leg_fk, lb_leg_fk]
    leg_forward_kinematics = fk_functions[leg_index]

    def cost_function(theta):
        current_position = leg_forward_kinematics(theta)
        l1 = np.abs(current_position - target_ee)
        cost = np.sum(l1**2)
        return cost, l1

    def gradient(theta, epsilon=1e-3):
        grad = np.zeros(3)
        for i in range(3):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            grad[i] = (cost_function(theta_plus)[0] - cost_function(theta_minus)[0]) / (2 * epsilon)
        return grad

    theta = np.array(initial_guess).astype(np.float64)
    learning_rate = 10
    max_iterations = 100
    tolerance = 1e-4

    cost_l = []
    for _ in range(max_iterations):
        grad = gradient(theta)
        theta -= learning_rate * grad
        cost, l1 = cost_function(theta)
        cost_l.append(cost)
        if l1.mean() < tolerance:
            break
    return theta
