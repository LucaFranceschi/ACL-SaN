import numpy as np
import os
import tqdm

if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk('train_outputs'):
        if len(filenames) > 1:
            for filename in tqdm.tqdm(filenames):
                if filename.endswith('.pkl'):
                    # out_dirpath = dirpath.replace('train_outputs', 'train_outputs2')
                    out_dirpath = dirpath
                    os.makedirs(out_dirpath, exist_ok=True)
                    arr = np.load(os.path.join(dirpath, filename), allow_pickle=True)
                    original_arr_shape = arr.shape
                    if len(arr.shape) == 2:
                        arr_mod = np.amax(arr, axis=1).astype(np.float16)
                        out_path_filename = os.path.join(out_dirpath, filename.replace('.pkl', ''))
                        np.save(out_path_filename, arr_mod)
                        os.remove(os.path.join(dirpath, filename))
                    elif len(arr.shape) > 2:
                        print(f'Array in {filename} has shape {arr.shape}! Not touching that...')