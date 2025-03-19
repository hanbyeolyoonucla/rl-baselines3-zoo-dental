from tqdm import tqdm
import yaml
import os

models = ['human']  # 'traction', 'coverage', 'human'

for model in models:
    model_dir = f'dental_env/demos_augmented/{model}_hdf5'
    dirlist = os.listdir(model_dir)
    print(f'{model}')

    with open(f'{model_dir}/{model}_data_stats.yml', 'w') as f:
        data_stat = {'data': [], 'CRE': 0, 'MIP': 0, 'total_collisions': 0, 'traverse_length': 0, 'traverse_angle': 0}

        # loop through all yml files
        n_data = 0
        for fname in tqdm(dirlist):
            if not fname.endswith('yml') or 'data_stats' in fname:
                continue
            with open(f'{model_dir}/{fname}', 'r') as s:
                data = yaml.safe_load(s)
                n_data += len(data)
                for info in data.values():
                    data_stat['CRE'] += info['CRE']
                    data_stat['MIP'] += info['MIP']
                    data_stat['total_collisions'] += info['total_collisions']
                    data_stat['traverse_length'] += info['traverse_length']
                    data_stat['traverse_angle'] += info['traverse_angle']
                data_stat['data'].append(fname)
        data_stat['CRE'] /= n_data
        data_stat['MIP'] /= n_data
        data_stat['total_collisions'] /= n_data
        data_stat['traverse_length'] /= n_data
        data_stat['traverse_angle'] /= n_data
        yaml.dump(data_stat, f)
