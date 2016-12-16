from multiprocessing import Process

import json
import os
import argparse
import subprocess

def collector(droidbot_out_path_list, apk_processor_path):
    """
    assemble the data from droidbot
    """
    pass


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    droidbot_out_path = os.path.abspath(config_json["droidbot_out_dir"])
    output_path = os.path.abspath(config_json["output_dir"])

    print droidbot_out_path
    print output_path


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="automated app testing script")
    parser.add_argument("-c", action="store", dest="config_json_path",
                        required=True, help="path to config json file")
    options = parser.parse_args()
    return options


def main():
    """
    the main function
    """
    opts = parse_args()
    run(opts.config_json_path)
    return


if __name__ == "__main__":
    main()
