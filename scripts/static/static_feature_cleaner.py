from multiprocessing import Process

import json
import os
import argparse
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# some init
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = stopwords.words("english")
first_cap_re = re.compile("(.)([A-Z][a-z]+)")
all_cap_re = re.compile("([a-z0-9])([A-Z])")


def id_convert(name):
    s1 = first_cap_re.sub(r"\1_\2", name)
    return all_cap_re.sub(r"\1_\2", s1).lower()


def clean_text(feature_text, dot_split=True):
    ret_obj = []
    if feature_text is None or len(feature_text) > 10000:
        return ret_obj

    if dot_split:
        feature_words = [x for x in " ".join(
            [" ".join(id_convert(x).split("_")) for x in feature_text.split(".")]
        ).split(" ") if len(x) > 0]
    else:
        feature_words = nltk.word_tokenize(feature_text)

    for feature_word in feature_words:
        if len(feature_word) < 3:
            continue
        try:
            new_word = stemmer.stem(wnl.lemmatize(feature_word.lower()))
            if new_word.encode("utf-8").isalpha() and new_word not in stop:
                ret_obj.append(new_word)
        except Exception as e:
            print e

    return ret_obj


def clean_obj_text(feature_obj, word_repeat, word_combine):
    ret_obj = {}

    for feature_key in feature_obj:
        feature_list = feature_obj[feature_key]
        ret_obj[feature_key] = []

        if feature_key in ["service", "activity", "provider", "receiver", "library"]:
            for feature_text in feature_list:
                ret_obj[feature_key] += clean_text(feature_text)

        elif feature_key in ["plurals", "string_arrays"]:
            for feature_bundle in feature_list:
                for feature_text in feature_bundle:
                    ret_obj[feature_key] += clean_text(feature_text, dot_split=False)

        elif feature_key in ["strings", "public"]:
            for feature_text in feature_list:
                ret_obj[feature_key] += clean_text(feature_text, dot_split=False)

        elif feature_key in ["class", "permission"]:
            ret_obj[feature_key] = feature_list

        if not word_repeat:
            ret_obj[feature_key] = list(set(ret_obj[feature_key]))

    if word_combine:
        combined_dict = {"combined": []}
        for feature_key in ret_obj:
            if feature_key not in ["class", "permission"]:
                combined_dict["combined"] += ret_obj[feature_key]
            else:
                combined_dict[feature_key] = ret_obj[feature_key]
        if not word_repeat:
            combined_dict["combined"] = list(set(combined_dict["combined"]))
        return combined_dict
    else:
        return ret_obj


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    feature_dir = os.path.abspath(config_json["feature_dir"])
    data_output_dir = os.path.abspath(config_json["data_output_dir"])
    word_repeat = config_json["word_repeat"]
    word_combine = config_json["word_combine"]

    feature_path_list = ["%s/%s" % (feature_dir, x)
                            for x in os.walk(feature_dir).next()[2]]

    count = 0
    for feature_path in feature_path_list:
        print feature_path
        print "%s/%s" % (str(count), str(len(feature_path_list)))
        count += 1
        with open(feature_path, "r") as feature_file:
            feature_obj = json.load(feature_file)
            if "class" in feature_obj and len(feature_obj["class"]) > 0:
                cleaned_obj = clean_obj_text(feature_obj, word_repeat, word_combine)
                with open("%s/%s" % (data_output_dir, feature_path.split("/")[-1]), "w") as output_file:
                    json.dump(cleaned_obj, output_file, indent=2)

def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="clean static features from apk")
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
