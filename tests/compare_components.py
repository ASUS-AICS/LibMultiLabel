import argparse
import os
import pickle

from nn.utils import get_names, compare


def main():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_dir", help="The directory to store the objects for the current branch")
    parser.add_argument("--master_dir", help="The directory to store the objects for the master branch")
    args = parser.parse_args()

    # Compare components
    names = get_names()
    for name in names:
        current_fp = open(os.path.join(args.current_dir, f"{name}.p"), "rb")
        master_fp = open(os.path.join(args.master_dir, f"{name}.p"), "rb")
        current_component = pickle.load(current_fp)
        master_component = pickle.load(master_fp)
        ret = compare(name, current_component, master_component)
        print(name, "PASSED" if ret else "FAILED")


if __name__ == "__main__":
    main()
