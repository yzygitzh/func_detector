from multiprocessing import Process
from xml.etree import ElementTree

import json
import os
import argparse
import subprocess

def collector_func(apk_path_list, class_dict, output_dir, apktool_path):
    """
    assemble the data from droidbot
    """
    for apk_path in apk_path_list:
        package_name = apk_path.split("/")[-1][:-len(".apk")]
        print package_name
        package_dict = {}
        try:
            subprocess.call([
                "java", "-jar", apktool_path, "d",
                "-o", "%s/%s" % (output_dir, package_name),
                "-s", apk_path
            ])
            # get permission list, activity/service/provider/receiver/lib names
            manifest_path = "%s/%s/AndroidManifest.xml" % (output_dir, package_name)
            manifest_root = ElementTree.parse(manifest_path).getroot()
            permission_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in manifest_root.getchildren() if "permission" in x.tag]))

            application_node = [x for x in manifest_root.getchildren() if "application" in x.tag][0]
            activity_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in application_node.getchildren() if "activity" in x.tag]))
            service_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in application_node.getchildren() if "service" in x.tag]))
            receiver_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in application_node.getchildren() if "receiver" in x.tag]))
            provider_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in application_node.getchildren() if "receiver" in x.tag]))
            library_list = list(set([x.attrib["{http://schemas.android.com/apk/res/android}name"]
                                        for x in application_node.getchildren() if "uses-library" in x.tag]))
        except Exception as e:
            print e

        try:
            # get public text id's and string/plural/arrays
            public_xml_path = "%s/%s/res/values/public.xml" % (output_dir, package_name)
            public_xml_root = ElementTree.parse(public_xml_path).getroot()
            public_string_list = list(set([(x.attrib["type"], x.attrib["name"])
                                            for x in public_xml_root.getchildren()
                                            if "APKTOOL_DUMMY" not in x.attrib["name"]]))
        except Exception as e:
            print e

        try:
            string_xml_path = "%s/%s/res/values/strings.xml" % (output_dir, package_name)
            string_xml_root = ElementTree.parse(string_xml_path).getroot()
            string_list = list(set([x.text for x in string_xml_root.getchildren()
                                    if "APKTOOL_DUMMY" not in x.attrib["name"]]))
        except Exception as e:
            print e

        try:
            plurals_xml_path = "%s/%s/res/values/plurals.xml" % (output_dir, package_name)
            plurals_xml_root = ElementTree.parse(plurals_xml_path).getroot()
            plurals_list = [[y.text for y in x.getchildren()]
                            for x in plurals_xml_root.getchildren()
                            if "APKTOOL_DUMMY" not in x.attrib["name"]]
        except Exception as e:
            print e

        try:
            str_array_xml_path = "%s/%s/res/values/arrays.xml" % (output_dir, package_name)
            str_array_xml_root = ElementTree.parse(str_array_xml_path).getroot()
            str_array_list = [[y.text for y in x.getchildren()]
                                for x in str_array_xml_root.getchildren()
                                if "string-array" in x.tag]
        except Exception as e:
            print e

        try:
            # do output
            package_dict["permission"] = permission_list
            package_dict["activity"] = activity_list
            package_dict["service"] = service_list
            package_dict["receiver"] = receiver_list
            package_dict["provider"] = provider_list
            package_dict["library"] = library_list
            package_dict["strings"] = string_list
            package_dict["plurals"] = plurals_list
            package_dict["string_arrays"] = str_array_list

            package_dict["class"] = []
            for class_name in class_dict:
                if package_name in class_dict[class_name]:
                    package_dict["class"].append(class_name)
        except Exception as e:
            print e

        try:
            with open("%s/%s.json" % (output_dir, package_name), "w") as output_file:
                json.dump(package_dict, output_file)
        except Exception as e:
            print e

        try:
            # delete decoded files
            subprocess.call([
                "rm", "-rf", "%s/%s" % (output_dir, package_name)
            ])
        except Exception as e:
            print e


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    apk_dir = os.path.abspath(config_json["apk_dir"])
    class_dir = os.path.abspath(config_json["class_dir"])
    output_dir = os.path.abspath(config_json["output_dir"])
    apktool_path = os.path.abspath(config_json["apktool_path"])
    process_num = config_json["process_num"]

    # build class dict
    class_dict = {}
    class_file_path_list = ["%s/%s" % (class_dir, x)
                            for x in os.walk(class_dir).next()[2]]
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
            class_dict, output_dir, apktool_path)))
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
