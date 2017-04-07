from multiprocessing import Process

import json
import os
import argparse

def collector_func(apk_data_path_list, output_path, exclude_activities):
    """
    assemble the data from droidbot
    """
    for apk_data_path in apk_data_path_list:
        trace_tag_list = [x[len("event_"):-len(".json")]
                          for x in os.walk("%s/events" % apk_data_path).next()[2]
                          if x.endswith("json")]
        trace_tag_list.sort()

        state_json_path_list = ["%s/states/%s" % (apk_data_path, x)
                                for x in os.walk("%s/states" % apk_data_path).next()[2]
                                if x.endswith("json")]

        # classify states into buckets by foreground_activity
        activity_bundle = {}
        state_tag_activity = {}
        for state_json_path in state_json_path_list:
            with open(state_json_path, "r") as state_json_file:
                state_json = json.load(state_json_file)

                state_activity = state_json["foreground_activity"]
                if state_activity in exclude_activities:
                    continue

                state_tag = state_json["tag"]
                state_tag_activity[state_tag] = state_activity

                if state_activity not in activity_bundle:
                    activity_bundle[state_activity] = {
                        "state_path_list":[], "screenshot_path_list": [],
                        "event_path_list":[], "event_trace_path_list": []}

                activity_bundle[state_activity]["state_path_list"].append(state_json_path)
                activity_bundle[state_activity]["screenshot_path_list"].append(
                    "%s/states/screenshot_%s.png" % (apk_data_path, state_tag))

        # fill in events and event_traces
        for i in range(len(trace_tag_list) - 1):
            trace_tag = trace_tag_list[i]
            relative_state_tag = trace_tag_list[i + 1]
            if relative_state_tag in state_tag_activity:
                state_activity = state_tag_activity[relative_state_tag]
                activity_bundle[state_activity]["event_path_list"].append(
                    "%s/events/event_%s.json" % (apk_data_path, trace_tag))
                activity_bundle[state_activity]["event_trace_path_list"].append(
                    "%s/events/event_%s.json" % (apk_data_path, trace_tag))

        for activity in activity_bundle:
            for log_type in activity_bundle[activity]:
                activity_bundle[activity][log_type].sort()

        with open("%s/%s.json" % (output_path, apk_data_path.split("/")[-1]), "w") as dump_file:
            json.dump(activity_bundle, dump_file, indent=2)


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    droidbot_out_path = os.path.abspath(config_json["droidbot_out_dir"])
    output_path = os.path.abspath(config_json["output_dir"])
    process_num = config_json["process_num"]
    exclude_activities = config_json["exclude_activities"]

    apk_data_path_list = ["%s/%s" % (droidbot_out_path, x)
                          for x in os.walk(droidbot_out_path).next()[1]]

    # start collectors
    package_name_trunk_len = (len(apk_data_path_list) + process_num - 1) / process_num
    collector_list = []
    for i in range(process_num):
        collector_list.append(Process(target=collector_func, args=(
            apk_data_path_list[i * package_name_trunk_len: (i + 1) * package_name_trunk_len],
            output_path, exclude_activities)))
        collector_list[-1].start()

    for collector in collector_list:
        collector.join()


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="merge multiple states and traces according to activities")
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
