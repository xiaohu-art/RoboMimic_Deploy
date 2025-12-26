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
from policy.motion.utils import quat_rotate_inverse, subtract_frame_transforms

class Motion(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_MOTION
        self.name_str = "skill_motion"
        self.counter_step = 0
        self.ref_motion_phase = 0

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
        observation = {}
        for in_key, in_shape in zip(self.in_keys, self.in_shapes):
            observation[in_key] = np.zeros(in_shape, dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)
        
        self.num_actions = outputs_result[self.out_keys.index("action")].shape[-1]
        self.prev_actions = np.zeros(self.num_actions, dtype=np.float32)

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
        self.ref_motion_phase = 0.
        self.motion_time = 0
        self.counter_step = 0

        self.action = np.zeros(self.num_actions)
        pass

    def get_robot_obs(self):
        root_pos_w = self.state_cmd.base_pos
        root_quat_w = self.state_cmd.base_quat
        root_angvel_b = quat_rotate_inverse(root_quat_w, self.state_cmd.ang_vel)
        projected_gravity_b = quat_rotate_inverse(root_quat_w, np.array([0., 0., -1.]))
        joint_pos = self.state_cmd.q[self.q_mjc2lab]
        joint_vel = self.state_cmd.dq[self.q_mjc2lab]
        prev_actions = self.prev_actions.copy()

        body_pos = self.state_cmd.body_pos
        
        obs_body_names = [
            "left_hip_pitch_link", "right_hip_pitch_link", 
            "left_knee_link", "right_knee_link", 
            "left_ankle_roll_link", "right_ankle_roll_link", 
            "left_shoulder_roll_link", "right_shoulder_roll_link", 
            "left_elbow_link", "right_elbow_link", 
            "left_wrist_yaw_link", "right_wrist_yaw_link"
        ]
        obs_body_idx = [body_names_mujoco.index(name) for name in obs_body_names]
        
        selected_body_pos = body_pos[obs_body_idx]
        body_pos_b = quat_rotate_inverse(root_quat_w, selected_body_pos - root_pos_w)
        
        return np.concatenate([
            root_quat_w,
            root_angvel_b,
            projected_gravity_b,
            joint_pos,
            joint_vel,
            prev_actions,
            body_pos_b.flatten()
        ]).astype(np.float32)

    def get_ref_motion_public_obs(self):
        frame_idx = self.counter_step % self.ref_q_lab.shape[0]
        ref_root_quat = self.ref_root_quat[frame_idx]

        kp_pos_mjc = self.state_cmd.body_pos
        kp_ori_mjc = self.state_cmd.body_ori
        kp_pos_lab = kp_pos_mjc[self.b_mjc2lab]
        kp_ori_lab = kp_ori_mjc[self.b_mjc2lab]

        ref_kp_pos_lab = self.ref_kp_pos_lab[frame_idx]
        ref_kp_ori_lab = self.ref_kp_ori_lab[frame_idx]

        pos, _ = subtract_frame_transforms(kp_pos_lab, kp_ori_lab, ref_kp_pos_lab, ref_kp_ori_lab)

        obs_body_names = [
            "pelvis",
            "left_hip_pitch_link", "right_hip_pitch_link", 
            "left_knee_link", "right_knee_link", 
            "left_ankle_roll_link", "right_ankle_roll_link", 
            "left_shoulder_roll_link", "right_shoulder_roll_link", 
            "left_elbow_link", "right_elbow_link", 
            "left_wrist_yaw_link", "right_wrist_yaw_link"
        ]

        obs_body_idx = [self.body_names_lab.index(name) for name in obs_body_names]
        pos = pos[obs_body_idx]
        return np.concatenate([
            ref_root_quat,
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
        self.prev_actions = self.action.copy()
        
        ref_q_lab = self.ref_q_lab[frame_idx]
        ref_q_mjc = ref_q_lab[self.q_lab2mjc]

        self.policy_output.actions = ref_q_mjc
        self.policy_output.kps[:] = self.kp_lab[self.q_lab2mjc]
        self.policy_output.kds[:] = self.kd_lab[self.q_lab2mjc]

        # Set root state
        self.policy_output.set_root_state = True
        self.policy_output.target_root_pos = self.ref_kp_pos_lab[frame_idx][0]
        self.policy_output.target_root_quat = self.ref_kp_ori_lab[frame_idx][0]
        self.policy_output.target_root_lin_vel = self.ref_kp_lin_vel_lab[frame_idx][0]
        self.policy_output.target_root_ang_vel = self.ref_kp_ang_vel_lab[frame_idx][0]

        self.counter_step += 1
        
    def exit(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.ref_motion_phase = 0.
        self.motion_time = 0
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