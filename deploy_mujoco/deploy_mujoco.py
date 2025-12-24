import sys
from pathlib import Path

from pynput import keyboard
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.keyboard_controller import KeyboardController



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    num_bodies = m.nbody - 1 # skip world body

    body_names_mjc, body_adrs = [], []
    for i in range(1, m.nbody): # skip the world body
        body = m.body(i)
        body_names_mjc.append(body.name)
        body_adrs.append(i)
    body_adrs = np.array(body_adrs)

    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints, num_bodies)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    keyboard = KeyboardController()
    Key = keyboard.Key
    print("Using Keyboard Control:")
    print("  Arrow Keys: Move")
    print("  Q/E: Turn")
    print("  p: Passive Mode")
    print("  l: Loco Mode")
    print("  z: Skill 1")
    print("  x: Skill 2")
    print("  c: Skill 3")
    print("  v: Skill 4")
    print("  b: Skill 5")

    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                if(keyboard.is_pressed(Key.esc)):
                    Running = False

                keyboard.update()
                
                # Check for commands
                if keyboard.is_released('p'):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                if keyboard.is_released(Key.enter):
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                if keyboard.is_released('l'):
                    state_cmd.skill_cmd = FSMCommand.LOCO
                if keyboard.is_released('z'):
                    state_cmd.skill_cmd = FSMCommand.SKILL_1
                if keyboard.is_released('x'):
                    state_cmd.skill_cmd = FSMCommand.SKILL_2
                if keyboard.is_released('c'):
                    state_cmd.skill_cmd = FSMCommand.SKILL_3
                if keyboard.is_released('v'):
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                if keyboard.is_released('b'):
                    state_cmd.skill_cmd = FSMCommand.SKILL_MOTION
                
                state_cmd.vel_cmd[0] = -keyboard.get_axis_value(1)
                state_cmd.vel_cmd[1] = -keyboard.get_axis_value(0)
                state_cmd.vel_cmd[2] = -keyboard.get_axis_value(3)
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    pos = d.qpos[0:3]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.base_pos = pos.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    state_cmd.body_pos = d.xpos[body_adrs].copy()
                    state_cmd.body_ori = d.xquat[body_adrs].copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
                    
                    if policy_output.set_root_state:
                        d.qpos[0:3] = policy_output.target_root_pos.copy()
                        d.qpos[3:7] = policy_output.target_root_quat.copy()
                        d.qvel[0:3] = policy_output.target_root_lin_vel.copy()
                        d.qvel[3:6] = policy_output.target_root_ang_vel.copy()
                        policy_output.set_root_state = False
            except ValueError as e:
                print(str(e))
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        