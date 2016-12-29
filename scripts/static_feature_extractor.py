from multiprocessing import Process

import json
import os
import argparse

def collector_func(apk_path_list, class_dict, output_path, apktool_path):
    """
    assemble the data from droidbot
    """
    for apk_path in apk_path_list:
        package_name = apk_path.split("/")[-1][:-len(".apk")]
        for class_name in class_dict:
            if package_name in class_dict[class_name]:
                print package_name


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    apk_dir = os.path.abspath(config_json["apk_dir"])
    class_dir = os.path.abspath(config_json["class_dir"])
    output_path = os.path.abspath(config_json["output_dir"])
    apktool_path = os.path.abspath(config_json["apktool_path"])
    process_num = config_json["process_num"]

    # build class dict
    class_dict = {}
    class_file_path_list = ["%s/%s" % (class_dir, x)
                            for x in os.walk(apk_dir).next()[2]]
    for class_file_path in class_file_path_list:
        with open(class_file_path, "r") as class_file:
            class_name = class_file_path.split("/")[-1][len("top_apps_in_"):-len(".txt")]
            package_names = [x for x in class_file.read().split('\n') if len(x) > 0]
            class_dict[class_name] = package_names

    apk_path_list = ["%s/%s" % (apk_dir, x)
                     for x in os.walk(apk_dir).next()[2]]

    # start collectors
    apk_path_trunk_len = (len(apk_path_list) + process_num - 1) / process_num
    collector_list = []
    for i in range(process_num):
        collector_list.append(Process(target=collector_func, args=(
            apk_path_list[i * apk_path_trunk_len: (i + 1) * apk_path_trunk_len],
            class_dict, output_path, apktool_path)))
        collector_list[-1].start()

    for collector in collector_list:
        collector.join()


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="extract static features from apk")
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
