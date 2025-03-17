import dental_env
import gymnasium as gym
import numpy as np
import os
from spatialmath import UnitQuaternion


def keyboard_to_action(key):
    act = np.array([0, 0, 0, 0, 0, 0])
    keymap = {'4': np.array([-1, 0, 0, 0, 0, 0]), '6': np.array([1, 0, 0, 0, 0, 0]),
              '1': np.array([0, -1, 0, 0, 0, 0]), '9': np.array([0, 1, 0, 0, 0, 0]),
              '2': np.array([0, 0, -1, 0, 0, 0]), '8': np.array([0, 0, 1, 0, 0, 0]),
              'a': np.array([0, 0, 0, -1, 0, 0]), 'd': np.array([0, 0, 0, 1, 0, 0]),
              'z': np.array([0, 0, 0, 0, -1, 0]), 'e': np.array([0, 0, 0, 0, 1, 0]),
              'x': np.array([0, 0, 0, 0, 0, -1]), 'w': np.array([0, 0, 0, 0, 0, 1]),
              }
    for c in key:
        act += keymap.get(c, np.array([0, 0, 0, 0, 0, 0]))
    return act


if __name__ == "__main__":

    # tooth
    tooth_dir = f'dental_env/labels_augmented'
    dirlist = os.listdir(tooth_dir)
    fname = dirlist[np.random.randint(0, len(dirlist))]
    tooth = fname[:-4]
    # tooth 2
    tooth = 'tooth_2_1.0_None_left_0_103_353_385'
    # tooth = 'tooth_2_1.0_None_right_2_268_235_394'
    # tooth = 'tooth_2_1.0_None_top_1_119_303_490'
    # tooth = 'tooth_2_1.0_None_top_3_228_317_483'
    # tooth = 'tooth_2_1.0_None_top_4_284_262_509'
    # tooth 3
    # tooth = 'tooth_3_1.0_None_right_2_231_179_453'
    # tooth = 'tooth_3_1.0_None_top_0_144_313_508'
    # tooth = 'tooth_3_1.0_None_top_1_227_258_489'
    # tooth 4
    # tooth = 'tooth_4_1.0_None_right_0_259_171_440'
    # tooth = 'tooth_4_1.0_None_top_1_142_349_479'
    # tooth = 'tooth_4_1.0_None_top_2_197_295_494'
    # tooth = 'tooth_4_1.0_None_top_3_190_229_502'
    # tooth 5
    # tooth = 'tooth_5_1.0_None_top_0_272_249_489'
    # tooth = 'tooth_5_1.0_None_top_1_118_180_484'
    # tooth = 'tooth_5_1.0_None_top_2_159_241_487'

    # Initialize gym environment
    env = gym.make("DentalEnvPCD-v0", render_mode="human", max_episode_steps=1000, tooth=tooth)
    state, info = env.reset(seed=42)

    total_reward = 0
    total_collisions = 0

    user_input = True
    itr = 0
    cutpath = [np.concatenate((info['position'], info['rotation']))]
    while user_input != 'n':
        # user keyboard input
        user_input = input("Keyboard input (n to stop): ")
        action = keyboard_to_action(user_input)
        state, reward, terminated, truncated, info = env.step(action)

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
        cutpath.append(np.concatenate((info['position'], info['rotation'])))

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
        print(f'traverse_angle: {traverse_length}')
        print(f'traverse_angle: {traverse_angle}')
        itr += 1

    env.close()
    np.savetxt(f'dental_env/demos_augmented/human/{tooth}.csv', cutpath)
