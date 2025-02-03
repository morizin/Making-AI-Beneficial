import pandas as pd, numpy as np, glob
import argparse, os
from termcolor import colored
os.system("color")
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_dir", type = str,help="directory where sub files are there with / at last")
    parser.add_argument('-d',"--data_dir", type = str,help="where the competition data is there with / at last")
    parser.add_argument('-o',"--out_file", type = str,help="output file name with .csv at last")
    return parser.parse_args()
args = arg()
main_dir = args.sub_dir
subs = {f'{main_dir}sub_sakt_model_v5.csv': 0.01, f'{main_dir}sub_sakt_model_v7.csv': 0.05,
        f'{main_dir}sub_sakt_model_v5_pseudo.csv': 0.015, f'{main_dir}sub_sakt_model_last_v1.csv': 0.0005,
        f'{main_dir}sub_sakt_model_v1_pseudo.csv': 0.001, f'{main_dir}sub_sakt_model_v1.csv': 0.0005,
        f'{main_dir}sub_sakt_model_v2.csv': 0.001, f'{main_dir}sub_sakt_model_v6.csv': 0.3045,
        f'{main_dir}sub_sakt_model_v6_adam.csv': 0.3045, f'{main_dir}sub_sakt_model_test_v1_pseudo.csv': 0.001,
        f'{main_dir}sub_sakt_model_v6_pseudo.csv': 0.3045, f'{main_dir}sub_sakt_model_last_v2.csv': 0.0005,
        f'{main_dir}sub_sakt_model_v3.csv': 0.005, f'{main_dir}sub_sakt_model_last_v3.csv': 0.002}

sub = pd.read_csv(f'{args.data_dir}sample_output.csv')

for sub_, weight in subs.items():
    sub['correct'] += np.array(pd.read_csv(sub_)['correct']* weight)
sub['correct'] = sub['correct']**2
sub.to_csv(f'{args.out_file}', index = False)
print(colored('Ensembled ', 'green'))
