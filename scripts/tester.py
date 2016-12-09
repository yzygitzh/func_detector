from multiprocessing import Process

import json
import os
import argparse
import subprocess

def tester(device_id, apk_path_list, interval, duration, event_policy, output_dir):
    """
    test apks on the assigned vm/device
    """
    for apk_path in apk_path_list:
        test_cmd = ("droidbot -d {device_id} -a {apk_path} -interval {interval} "
                    "-duration {duration} -event {event_policy} -o {output_dir}").format(
                        device_id=device_id,
                        apk_path=apk_path,
                        interval=interval,
                        duration=duration,
                        event_policy=event_policy,
                        output_dir="%s/%s" % (output_dir, apk_path.split("/")[-1][:-len(".apk")]))
        subprocess.call(test_cmd.split())


def run(config_json_path):
    """
    parse config file and assign work to multiple vm/device's
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    device_id_list = config_json["device_id_list"]
    device_num = len(device_id_list)

    apk_directory = os.path.abspath(config_json["apk_directory"])
    apk_path_list = ["%s/%s" % (apk_directory, x) for x in
                     [x for x in os.walk(apk_directory).next()[2] if x.endswith("apk")]]

    event_policy = config_json["event_policy"]
    interval = config_json["interval"]
    duration = config_json["duration"]
    output_dir = os.path.abspath(config_json["output_dir"])

    # start testers
    apk_trunk_len = (len(apk_path_list) + device_num - 1) / device_num
    testor_list = []
    for i in range(device_num):
        testor_list.append(Process(target=tester, args=(
            device_id_list[i], apk_path_list[i * apk_trunk_len: (i + 1) * apk_trunk_len],
            interval, duration, event_policy, output_dir)))
        testor_list[-1].start()

    for i in range(device_num):
        testor_list[i].join()


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
