import os, sys
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default="single", type=str, help="analyze type")
    parser.add_argument('-d', '--data_dir', default="~/v2x/", type=str, help="data directory")

    args = parser.parse_args()




if __name__ == '__main__':
    main()