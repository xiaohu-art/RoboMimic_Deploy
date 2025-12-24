from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints, num_bodies):
        # root state
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_quat = np.zeros(4, dtype=np.float32)
        # joint state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.dq = np.zeros(num_joints, dtype=np.float32)
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        # gravity orientation
        self.gravity_ori = np.array([0., 0., 1.])
        # angular velocity
        self.ang_vel = np.zeros(3)
        # body state
        self.body_pos = np.zeros((num_bodies, 3), dtype=np.float32)
        self.body_ori = np.zeros((num_bodies, 4), dtype=np.float32)
        # joy cmd
        self.vel_cmd = np.zeros(3)
        self.skill_cmd = FSMCommand.INVALID
        # skill change cmd
        # self.skill_set = FSMCommand.SKILL_1

class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)
        
        self.set_root_state = False
        self.target_root_pos = np.zeros(3, dtype=np.float32)
        self.target_root_quat = np.zeros(4, dtype=np.float32)
        self.target_root_lin_vel = np.zeros(3, dtype=np.float32)
        self.target_root_ang_vel = np.zeros(3, dtype=np.float32)
        