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
from policy.motion.utils import *
from policy.motion.observation import *

class Motion(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_MOTION
        self.name_str = "skill_motion"
        self.counter_step = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.onnx_path = os.path.join(current_dir, "model", "motion.onnx")
        self.motion_file = os.path.join(current_dir, "data", "sfu_29dof.pkl")
        
        asset_config_path = os.path.join(current_dir, "config", "asset_meta.json")
        with open(asset_config_path, "r") as f:
            asset_config = json.load(f)
            self.body_names_lab = asset_config["body_names_isaac"]
            self.joint_names_lab = asset_config["joint_names_isaac"]
            self.default_joint_pos_lab = np.array(asset_config["default_joint_pos"], dtype=np.float32)
            self.kp_lab = np.array(asset_config["stiffness"], dtype=np.float32)
            self.kd_lab = np.array(asset_config["damping"], dtype=np.float32)

        policy_config_path = os.path.join(current_dir, "config", "policy.json")
        with open(policy_config_path, "r") as f:
            policy_config = json.load(f)
            self.action_scaling = 0.5
            self.in_keys = policy_config["in_keys"]
            self.out_keys = policy_config["out_keys"]
            self.in_shapes = policy_config["in_shapes"][0]

        self.q_lab2mjc = np.array([self.joint_names_lab.index(name) for name in joint_names_mujoco])
        self.q_mjc2lab = np.array([joint_names_mujoco.index(name) for name in self.joint_names_lab])

        self.b_lab2mjc = np.array([self.body_names_lab.index(name) for name in body_names_mujoco])
        self.b_mjc2lab = np.array([body_names_mujoco.index(name) for name in self.body_names_lab])

        self.joint_ids = np.array([self.joint_names_lab.index(name) for name in action_joint_names_lab])

        self.load_motion()

        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)

        self.articulation = Articulation(self.state_cmd, self.q_mjc2lab, self.b_mjc2lab)

        self.robot_obs_list = [
            root_quat_w(self.articulation),
            root_angvel_b(self.articulation),
            projected_gravity_b(self.articulation),
            joint_pos(self.articulation),
            joint_vel(self.articulation),
            body_pos_b(self.articulation, 
                [   "left_hip_pitch_link", "right_hip_pitch_link", 
                    "left_knee_link", "right_knee_link", 
                    "left_ankle_roll_link", "right_ankle_roll_link", 
                    "left_shoulder_roll_link", "right_shoulder_roll_link", 
                    "left_elbow_link", "right_elbow_link", 
                    "left_wrist_yaw_link", "right_wrist_yaw_link"]
            ),
        ]

    def load_motion(self):
        with open(self.motion_file, "rb") as f:
            motions = joblib.load(f)
            motion_name = list(motions.keys())[0]
            print("Loading motion: ", motion_name)
            motion = motions[motion_name]
            self.fps = motion["fps"]
            self.ref_q_lab = motion["joint_pos"]        # (T, num_joints)
            self.ref_dq_lab = motion["joint_vel"]        # (T, num_joints)
            self.ref_kp_pos_lab = motion["body_pos_w"]    # (T, num_bodies, 3)
            self.ref_kp_ori_lab = motion["body_quat_w"]    # (T, num_bodies, 4)
            self.ref_kp_lin_vel_lab = motion["body_lin_vel_w"]    # (T, num_bodies, 3)
            self.ref_kp_ang_vel_lab = motion["body_ang_vel_w"]    # (T, num_bodies, 3)
            self.ref_root_pos = self.ref_kp_pos_lab[:, 0]    # (T, 3)
            self.ref_root_quat = self.ref_kp_ori_lab[:, 0]    # (T, 4)

    def enter(self):
        self.motion_time = 0
        self.counter_step = 0

        # Adjust reference motion based on current robot root state
        root_pos_w = self.state_cmd.base_pos.copy()
        root_quat_w = self.state_cmd.base_quat.copy()
        
        # 1. Calculate Yaw offset
        robot_yaw_q = yaw_quat(root_quat_w)
        ref_start_yaw_q = yaw_quat(self.ref_root_quat[0])
        delta_yaw_q = quat_mul(robot_yaw_q, quat_inv(ref_start_yaw_q))

        # 2. Transform all reference data
        # Position transformation: P_new = P_robot_start + R_delta_yaw * (P_ref - P_ref_start)
        ref_start_pos = self.ref_kp_pos_lab[0, 0].copy()
        
        # We transform all bodies at once
        # (T, num_bodies, 3)
        rel_pos = self.ref_kp_pos_lab - ref_start_pos
        self.ref_kp_pos_lab = root_pos_w + quat_apply(delta_yaw_q, rel_pos)
        
        # Orientation transformation: Q_new = delta_yaw * Q_ref
        # (T, num_bodies, 4)
        T, N, _ = self.ref_kp_ori_lab.shape
        self.ref_kp_ori_lab = quat_mul(
            np.tile(delta_yaw_q, (T, N, 1)), 
            self.ref_kp_ori_lab
        )
        
        # Velocity transformation: V_new = delta_yaw * V_ref
        # (T, num_bodies, 3)
        self.ref_kp_lin_vel_lab = quat_apply(delta_yaw_q, self.ref_kp_lin_vel_lab)
        self.ref_kp_ang_vel_lab = quat_apply(delta_yaw_q, self.ref_kp_ang_vel_lab)

        # Update root state helpers
        self.ref_root_pos = self.ref_kp_pos_lab[:, 0]    # (T, 3)
        self.ref_root_quat = self.ref_kp_ori_lab[:, 0]    # (T, 4)

        observation = {}
        for in_key, in_shape in zip(self.in_keys, self.in_shapes):
            observation[in_key] = np.zeros(in_shape, dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)
        self.actions = outputs_result[self.out_keys.index("action")].squeeze(0)
        pass

    def get_robot_obs(self):
        obs_data = [o() for o in self.robot_obs_list]
        return np.concatenate([
            *obs_data[:-1],    # root_quat to joint_vel
            self.actions,
            obs_data[-1]       # body_pos_b
        ]).astype(np.float32)

    def get_ref_motion_public_obs(self):
        frame_idx = self.counter_step % self.ref_q_lab.shape[0]
        ref_root_quat_w = self.ref_root_quat[frame_idx]

        body_indices, body_names = resolve_matching_names(
            [   "pelvis",
                "left_hip_pitch_link", "right_hip_pitch_link", 
                "left_knee_link", "right_knee_link", 
                "left_ankle_roll_link", "right_ankle_roll_link", 
                "left_shoulder_roll_link", "right_shoulder_roll_link", 
                "left_elbow_link", "right_elbow_link", 
                "left_wrist_yaw_link", "right_wrist_yaw_link"
            ],
            body_names_isaac
        )

        ref_kp_pos_w = self.ref_kp_pos_lab[frame_idx][body_indices]
        ref_kp_ori_w = self.ref_kp_ori_lab[frame_idx][body_indices]

        body_pos_w = self.articulation.body_pos_w[body_indices]
        body_quat_w = self.articulation.body_quat_w[body_indices]

        pos, _ = subtract_frame_transforms(body_pos_w, body_quat_w, ref_kp_pos_w, ref_kp_ori_w)
        return np.concatenate([
            ref_root_quat_w,
            pos.flatten()
        ]).astype(np.float32)

    def run(self):
        frame_idx = self.counter_step % self.ref_q_lab.shape[0]

        observation = {}
        for i, in_key in enumerate(self.in_keys):
            get_obs_fn = getattr(self, f"get_{in_key}_obs", None)
            if get_obs_fn is not None:
                observation[in_key] = get_obs_fn().reshape(1, -1)
            else:
                observation[in_key] = np.zeros(self.in_shapes[i], dtype=np.float32)
        
        outputs_result = self.ort_session.run(None, observation)
        self.action = outputs_result[self.out_keys.index("action")].squeeze(0)
        
        ref_q_lab = self.ref_q_lab[frame_idx]
        ref_q_mjc = ref_q_lab[self.q_lab2mjc]

        self.policy_output.actions = ref_q_mjc
        self.policy_output.kps[:] = self.kp_lab[self.q_lab2mjc]
        self.policy_output.kds[:] = self.kd_lab[self.q_lab2mjc]

        # Set root state
        self.policy_output.set_root_state = True
        self.policy_output.target_root_pos = self.ref_root_pos[frame_idx]
        self.policy_output.target_root_quat = self.ref_root_quat[frame_idx]
        self.policy_output.target_root_lin_vel = self.ref_kp_lin_vel_lab[frame_idx][0]
        self.policy_output.target_root_ang_vel = self.ref_kp_ang_vel_lab[frame_idx][0]

        self.counter_step += 1
        
    def exit(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.counter_step = 0
        print()
        
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_MOTION

action_joint_names_lab = [
    'left_hip_pitch_joint', 
    'right_hip_pitch_joint', 
    'waist_yaw_joint', 
    'left_hip_roll_joint', 
    'right_hip_roll_joint', 
    'left_hip_yaw_joint', 
    'right_hip_yaw_joint', 
    'left_knee_joint', 
    'right_knee_joint', 
    'left_shoulder_pitch_joint', 
    'right_shoulder_pitch_joint', 
    'left_ankle_pitch_joint', 
    'right_ankle_pitch_joint', 
    'left_shoulder_roll_joint', 
    'right_shoulder_roll_joint', 
    'left_ankle_roll_joint', 
    'right_ankle_roll_joint', 
    'left_shoulder_yaw_joint', 
    'right_shoulder_yaw_joint', 
    'left_elbow_joint', 
    'right_elbow_joint', 
    'left_wrist_roll_joint', 
    'right_wrist_roll_joint', 
    'left_wrist_pitch_joint', 
    'right_wrist_pitch_joint', 
    'left_wrist_yaw_joint', 
    'right_wrist_yaw_joint'
]

joint_names_mujoco = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

body_names_mujoco = [
    'pelvis', 
    'left_hip_pitch_link', 
    'left_hip_roll_link', 
    'left_hip_yaw_link', 
    'left_knee_link', 
    'left_ankle_pitch_link', 
    'left_ankle_roll_link', 
    'right_hip_pitch_link', 
    'right_hip_roll_link', 
    'right_hip_yaw_link', 
    'right_knee_link', 
    'right_ankle_pitch_link', 
    'right_ankle_roll_link', 
    'waist_yaw_link', 
    'waist_roll_link', 
    'torso_link', 
    'left_shoulder_pitch_link', 
    'left_shoulder_roll_link', 
    'left_shoulder_yaw_link', 
    'left_elbow_link', 
    'left_wrist_roll_link', 
    'left_wrist_pitch_link', 
    'left_wrist_yaw_link', 
    'right_shoulder_pitch_link', 
    'right_shoulder_roll_link', 
    'right_shoulder_yaw_link', 
    'right_elbow_link', 
    'right_wrist_roll_link', 
    'right_wrist_pitch_link', 
    'right_wrist_yaw_link'
]