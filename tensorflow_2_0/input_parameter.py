import argparse

def get_args():
    parser = argparse.ArgumentParser(prog="args",
                                     description="Singing-Voice-Seperation", add_help=True)
    parser.add_argument('-d', '--DATADIR', help='data diractory.', required=True)
    return parser.parse_args()
