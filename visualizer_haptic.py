import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from ibrl_td3.bc_policies import CustomActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from hyperparams.python.ppo_config import hyperparams
from gymnasium.wrappers import TransformReward
from ibrl_td3.ibrl import IBRL
from stable_baselines3 import TD3
from ibrl_custom_td3 import CustomTD3
import os
from spatialmath import UnitQuaternion
from traction import Traction
import time
import socket
import struct

HOST = '127.0.0.1'
PORT = 65432

if __name__ == "__main__":

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        # Select random tooth
        # tooth_dir = f'dental_env/demos_augmented/{model}'
        # dirlist = os.listdir(tooth_dir)
        # fname = dirlist[np.random.randint(1, len(dirlist))]
        # tooth = fname[:-4]

        # Select specific tooth
        tooth = 'tooth_3_1.0_None_top_0_144_313_508'  # MIP: 7.494094488188976
        # tooth='tooth_3_1.0_None_top_1_227_258_489'  # MIP: 7.002118644067797
        # tooth='tooth_2_1.0_None_top_1_119_303_490'  # MIP: 5.384761904761905 traverse: 14.8/67.0
        # tooth='tooth_2_1.0_None_top_3_228_317_483'  # MIP: 7.802105263157895
        # tooth='tooth_2_1.0_None_top_4_284_262_509'  # MIP: 9.994252873563218
        # tooth='tooth_4_1.0_None_top_1_142_349_479'  # MIP: 4.699074074074074
        # tooth='tooth_4_1.0_None_top_2_197_295_494'  # MIP: 5.74375
        # tooth='tooth_4_1.0_None_top_3_190_229_502'  # MIP: 8.657777777777778
        # tooth='tooth_5_1.0_None_top_0_272_249_489'  # MIP: 13.678571428571429
        # tooth='tooth_5_1.0_None_top_1_118_219_484'  # MIP: 11.854961832061068
        # tooth='tooth_5_1.0_None_top_2_159_241_487'  # MIP: 7.428571428571429

        # Initialize gym environment
        env = gym.make("DentalEnvPCD-v0", render_mode="human", max_episode_steps=None, tooth=tooth, force_feedback=True)
        state, info = env.reset(seed=42)

        total_reward = 0
        total_collisions = 0
        itr = 0
        cutpath = []

        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by ', addr)
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    dx, dy, dz, r, p, y = struct.unpack('ffffff', data)
                    print(f"Received position: dx={dx}, dy={dy}, dz={dz}")
                    print(f"Received rotation: dx={r}, dy={p}, dz={y}")
                    action = [dx, dy, dz, r, p, y]
                    cur_time = time.time()
                    state, reward, terminated, truncated, info = env.step(action)
                    print(f"colapsed time for each step {time.time() - cur_time}")

                    force = info["force_feedback"]
                    response = struct.pack('fff', force[0], force[1], force[2])
                    conn.sendall(response)

                    total_reward = total_reward + reward
                    decay_removal = info['decay_removal']
                    enamel_damage = info['enamel_damage']
                    dentin_damage = info['dentin_damage']
                    total_collisions = total_collisions + info['is_collision']
                    success = info['is_success']
                    tooth_name = info['tooth']
                    cre = info['CRE']
                    mip = info['MIP']
                    traverse_length = info['traverse_length']
                    traverse_angle = info['traverse_angle']
                    pos = info['position']
                    rot = info['rotation']

                    print(f'-------iteration: {itr}-------')
                    print(f'tooth: {tooth_name}')
                    print(f'success: {success}')
                    print(f'total_reward: {total_reward}')
                    print(f'decay_removal [%]: {decay_removal}')
                    print(f'enamel_damage [voxel]: {enamel_damage}')
                    print(f'dentin_damage [voxel]: {dentin_damage}')
                    print(f'CRE: {cre}')
                    print(f'MIP: {mip}')
                    print(f'total_collisions: {total_collisions}')
                    print(f'traverse_length: {traverse_length}')
                    print(f'traverse_angle: {traverse_angle}')
                    print(f'position: {pos}')
                    print(f'rotation: {rot}')

                    cutpath.append(np.concatenate((pos, rot)))
                    itr += 1

                    if terminated or truncated:
                        env.close()
                        break

                except ConnectionResetError:
                    print("Client disconnected.")
                    break
            np.savetxt(f'cutpath/human_data/{tooth}_haptic_1.txt', cutpath)
            env.close()






