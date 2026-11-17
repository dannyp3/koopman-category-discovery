"""
Generate a 3-dimensional dynamical systems dataset and save it to
<repo>/data/3-dimensional-systems/dataset_<num_cats>_class_<num_samples>_<samples_name>.pkl

Script version of scripts/create_high_dim_systems.ipynb so the data can be
regenerated from the command line (e.g. on a fresh Linux machine).

Example:
    python scripts/create_dataset.py --num-samples 50 --systems lorenz rossler chen halvorsen
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy

from kcm.dynamical_systems import create_three_dimensional_dataset
from kcm.koopman_category_model import DYNAMICAL_SYSTEMS


REPO_ROOT = Path(__file__).resolve().parents[1]

THREE_DIM_SYSTEMS = ['lorenz', 'rossler', 'chen', 'halvorsen', 'sprott_a', 'tigan', 'cross_coupled']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--systems', nargs='+', default=THREE_DIM_SYSTEMS[:4], choices=THREE_DIM_SYSTEMS,
                        help='3-D systems to include (order defines the class/target index)')
    parser.add_argument('--num-samples', type=int, default=100, help='Trajectories per system')
    parser.add_argument('--n', type=int, default=3000, help='Time points per trajectory')
    parser.add_argument('--begin', type=float, default=0)
    parser.add_argument('--end', type=float, default=200)
    parser.add_argument('--noise-std', type=float, default=0.01, help='Noise as a fraction of per-channel signal std')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    systems = {name: DYNAMICAL_SYSTEMS[name] for name in args.systems}

    print({'seed': args.seed, 'scipy_version': scipy.__version__, 'numpy_version': np.__version__})
    dataset = create_three_dimensional_dataset(args.n, systems, args.begin, args.end,
                                               args.num_samples, args.noise_std, seed=args.seed)

    samples_name = f'noisy_{args.noise_std}_samples' if args.noise_std > 0 else 'samples'
    path = REPO_ROOT / 'data' / '3-dimensional-systems' / f"dataset_{len(systems)}_class_{args.num_samples}_{samples_name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Saved dataset to {path}')


if __name__ == '__main__':
    main()
