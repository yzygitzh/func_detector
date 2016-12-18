import json
import os
import argparse

from pymongo import MongoClient

def run(config_json_path):
    """
    parse config file and manipulate collections
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    host = config_json["host"]
    port = config_json["port"]
    apk_bundle_list_path = os.path.abspath(config_json["apk_bundle_list_path"])

    apk_bundle_path_list = ["%s/%s" % (apk_bundle_list_path, x)
                            for x in os.walk(apk_bundle_list_path).next()[2]]

    client = MongoClient(host=host, port=port)
    db = client[config_json["db"]]
    db[config_json["unknown_collection"]].drop()
    # db[config_json["known_collection"]].drop()

    for apk_bundle_path in apk_bundle_path_list:
        with open(apk_bundle_path, "r") as apk_bundle_file:
            apk_bundle = json.load(apk_bundle_file)
            for activity in apk_bundle:
                mongo_bundle = {"activity": activity, "info": apk_bundle[activity]}
                db[config_json["unknown_collection"]].insert_one(mongo_bundle)


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="reset mongodb, fill in data")
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
