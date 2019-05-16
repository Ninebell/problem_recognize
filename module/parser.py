import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", help="increase output verbosity")


def img_path():
    args = parser.parse_args()
    if args.path:
        return args.path
    else:
        return "./test0.PNG"

