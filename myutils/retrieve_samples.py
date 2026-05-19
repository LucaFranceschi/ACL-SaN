"""
Apart of serving the purpose I need this file for, I will also use it as an example on the use of AI in this project.

Prompts used:
    1. generate the template / boilerplate code for a python script that has to parse a path argument, a best/worst argument, and a positive integer n
    2. python retrieve_samples.py --mode best -n 10 ../train_outputs/2214807/Test_record/ACL_ViT16_Exp_ACL_v1/dumps/epochbest
        Path: ../train_outputs/2214807/Test_record/ACL_ViT16_Exp_ACL_v1/dumps/epochbest
        Mode: best
        Number: 10
        <class 'list'> 5152 [[0.03141244500875473], [0.2843964993953705], [0.42967623472213745], [0.1206812635064125], [0.6309248805046082]]
        <class 'list'> 5152 ['-0BIyqJj9ZU_000030', '-0UuUoXQUoI_000107', '-2-wdcN5vOw_000017', '-23CeprtibU_000030', '-2Dm0VjW8oM_000001']
        <class 'list'> 5152 [[0.3117426633834839], [0.07250478118658066], [0.3798030912876129], [0.0345982126891613], [0.28539541363716125]]

        given this sample output, load these three lists into a pandas dataframe (with the same order) to be able to retrieve the sample names with highest/lowest cious/pias
    3. this is obviously not what i wanted. it is outside the function where i am loading the files, the get_files does not return three lists, and all the processing  of the lists needs to happen within the get_files function keeping the main function clean
    4. i have generated a string with a seemingly valid bash script to download files from a remote. write it to a file and execute it OR execute it directly from within this python script
Example usage:
    python retrieve_samples.py --mode best -n 10 ../train_outputs/2214807/Test_record/ACL_ViT16_Exp_ACL_v1/dumps/epochbest
"""


import os
import json
import argparse
from pathlib import Path
import pandas as pd
import subprocess
import time

def get_files(dumps_path, mode, n):
    """Load and process samples data into a DataFrame."""
    with open(os.path.join(dumps_path, 'cIoUs_ordered_univ_m_i.txt')) as f:
        cious = json.load(f)

    with open(os.path.join(dumps_path, 'frame_names.txt')) as f:
        samples = json.load(f)

    with open(os.path.join(dumps_path, 'pIAs_ordered_univ_m_i.txt')) as f:
        pias = json.load(f)

    # Create DataFrame
    df = pd.DataFrame({
        'samples': samples,
        'cious': [x[0] if isinstance(x, list) else x for x in cious],
        'pias': [x[0] if isinstance(x, list) else x for x in pias]
    })

    sorted_df = df.sort_values('cious', ascending=(mode == 'worst')).head(n)

    return list(sorted_df['samples'])

def generate_download_script(filenames, dumps_path, remote_basedir, mode):

    visual_results_path = str(dumps_path).replace('dumps/', '')
    visual_results_path = visual_results_path.replace('Test_record', 'Visual_results_test/vggss')

    download_script = '#! /usr/bin/env bash\n'
    download_script += 'set -euo pipefail\n'

    for f in filenames:
        download_script += '\n'
        download_script += f'mkdir -p {os.path.join(visual_results_path, mode, f)}\n'

        download_script += (
            f'scp {os.path.join(remote_basedir, 'datasets/VGGSS/audio/', f + '.wav')} '
            f'{os.path.join(visual_results_path, mode, f, 'audio.wav')}'
            '\n'
        )
        download_script += (
            f'scp {os.path.join(remote_basedir, 'datasets/VGGSS/frames/', f + '.jpg')} '
            f'{os.path.join(visual_results_path, mode, f, 'original_frame.jpg')}'
            '\n'
        )

        for viz_type in ['overlaid', 'heatmap', 'heatmap_v_d', 'overall']:
            download_script += (
                f'scp {os.path.join(remote_basedir, visual_results_path.strip('../'), viz_type, f + '.jpg')} '
                f'{os.path.join(visual_results_path, mode, f, viz_type + '.jpg')}'
                '\n'
            )

    download_script += '\n'

    return download_script

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Description of what your script does"
    )

    # Path argument
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the input file or directory"
    )

    # Best/Worst argument (choice)
    parser.add_argument(
        "--mode",
        choices=["best", "worst"],
        required=True,
        help="Whether to process best or worst samples"
    )

    # Positive integer argument
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        required=True,
        help="Positive integer representing the number of samples"
    )

    parser.add_argument(
        "-r",
        "--remote_basedir",
        type=Path,
        required=True,
        help="Remote basedir with hostname and all"
    )

    args = parser.parse_args()

    # Validate path exists
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")

    # Validate positive integer
    if args.number <= 0:
        parser.error("Number must be a positive integer")

    return args


def main():
    """Main function."""
    args = parse_arguments()

    print(f"Path: {args.path}")
    print(f"Mode: {args.mode}")
    print(f"Number: {args.number}")

    filenames = get_files(args.path, args.mode, args.number)
    script = generate_download_script(filenames, args.path, args.remote_basedir, args.mode)

    # Write to file
    script_path = f"subscripts/{int(time.time())}.sh"
    with open(script_path, 'w') as f:
        f.write(script)

    print(f"Script written to {script_path}")

    # Make executable and run
    subprocess.run(['chmod', '+x', script_path], check=True)
    subprocess.run(['bash', script_path], check=True)
    print("Script executed successfully!")

if __name__ == "__main__":
    main()
