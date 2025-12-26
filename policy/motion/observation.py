from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os
import json
import joblib
from common.utils import get_gravity_orientation
from policy.motion.utils import *

class Articulation:
    def __init__(
        self,
        state_cmd:StateAndCmd,
        q_mjc2lab:np.ndarray,
        b_mjc2lab:np.ndarray,
    ):
        self.state_cmd = state_cmd
        self.q_mjc2lab = q_mjc2lab
        self.b_mjc2lab = b_mjc2lab

    @property
    def root_pos_w(self):
        return self.state_cmd.base_pos

    @property
    def root_quat_w(self):
        return self.state_cmd.base_quat

    @property
    def root_ang_vel_w(self):
        return self.state_cmd.ang_vel

    @property
    def joint_pos(self):
        return self.state_cmd.q[self.q_mjc2lab]

    @property
    def joint_vel(self):
        return self.state_cmd.dq[self.q_mjc2lab]

    @property
    def body_pos_w(self):
        return self.state_cmd.body_pos[self.b_mjc2lab]

    @property
    def body_quat_w(self):
        return self.state_cmd.body_ori[self.b_mjc2lab]

class Observation:
    def __init__(
        self,
        articulation:Articulation,
    ):
        self.articulation = articulation

    def __call__(self):
        pass

class root_quat_w(Observation):
    def __call__(self):
        return self.articulation.root_quat_w

class root_angvel_b(Observation):
    def __call__(self):
        return quat_rotate_inverse(self.articulation.root_quat_w, self.articulation.root_ang_vel_w)

class projected_gravity_b(Observation):
    def __call__(self):
        return quat_rotate_inverse(self.articulation.root_quat_w, np.array([0., 0., -1.]))

class joint_pos(Observation):
    def __call__(self):
        return self.articulation.joint_pos

class joint_vel(Observation):
    def __call__(self):
        return self.articulation.joint_vel

class body_pos_w(Observation):
    def __init__(self, articulation:Articulation, body_names):
        super().__init__(articulation)
        self.body_indices, self.body_names = resolve_matching_names(body_names, body_names_isaac)

    def __call__(self):
        return self.articulation.body_pos_w[self.body_indices]

class body_quat_w(Observation):
    def __init__(self, articulation:Articulation, body_names):
        super().__init__(articulation)
        self.body_indices, self.body_names = resolve_matching_names(body_names, body_names_isaac)

    def __call__(self):
        return self.articulation.body_quat_w[self.body_indices]

class body_pos_b(Observation):
    def __init__(self, articulation:Articulation, body_names):
        super().__init__(articulation)
        self.body_indices, self.body_names = resolve_matching_names(body_names, body_names_isaac)

    def __call__(self):
        root_pos_w = self.articulation.root_pos_w
        root_quat_w = self.articulation.root_quat_w
        body_pos_w = self.articulation.body_pos_w[self.body_indices]
        body_pos_b = quat_rotate_inverse(root_quat_w, body_pos_w - root_pos_w)
        return body_pos_b.flatten()