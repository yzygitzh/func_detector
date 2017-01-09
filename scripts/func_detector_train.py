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
stop = stopwords.words('english')
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def id_convert(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()


def clean_text(feature_text, dot_split=True):
    ret_obj = set()
    if feature_text is None or len(feature_text) > 10000:
        return ret_obj

    if dot_split:
        feature_words = [x for x in " ".join(
            [" ".join(id_convert(x).split("_")) for x in feature_text.split(".")]
        ).split(" ") if len(x) > 0]
    else:
        feature_words = nltk.word_tokenize(feature_text)

    for feature_word in feature_words:
        try:
            new_word = stemmer.stem(wnl.lemmatize(feature_word.lower()))
            if new_word.encode("utf-8").isalpha() and new_word not in stop:
                ret_obj.add(new_word)
        except Exception as e:
            print e
    return ret_obj


def clean_obj_text(feature_obj):
    ret_obj = {
        "service": set(),
        "activity": set(),
        "provider": set(),
        "receiver": set(),
        "library": set(),
        "permission": feature_obj["permission"],
        "class": feature_obj["class"],
        "strings": set(),
        "plurals": set(),
        "string_arrays": set(),
        "public": set()
    }
    for feature_key in feature_obj:
        feature_list = feature_obj[feature_key]
        if feature_key in ["service", "activity", "provider", "receiver", "library"]:
            for feature_text in feature_list:
                ret_obj[feature_key] = ret_obj[feature_key].union(clean_text(feature_text))
        elif feature_key in ["plurals", "string_arrays"]:
            for feature_bundle in feature_list:
                for feature_text in feature_bundle:
                    ret_obj[feature_key] = ret_obj[feature_key].union(
                        clean_text(feature_text, dot_split=False))
        elif feature_key in ["strings", "public"]:
            for feature_text in feature_list:
                ret_obj[feature_key] = ret_obj[feature_key].union(
                    clean_text(feature_text, dot_split=False))

    for feature_key in ["service", "activity", "provider", "receiver", "library",
                        "plurals", "string_arrays", "strings"]:
        ret_obj[feature_key] = list(ret_obj[feature_key])

    return ret_obj


def build_word_bag(sample_list, config_json):
    # ret wb["WEATHER"]["activity"] == ["weather", "rain", ...]

    # select features in different dimensions by calculating mi
    # using sklearn's method

    # calc label set
    # calc the whole word set for feature dim's
    label_set = set()
    dim_word_set = {
        "manifest": set(),
        "string": set(),
        "public": set(),
        "permission": set()
    }

    count = 0
    for sample in sample_list:
        print "%s/%s" % (str(count), str(len(sample_list)))
        count += 1
        label_set = label_set.union(set(sample["class"]))
        for dim_name in sample:
            if dim_name in ["class"]:
                continue
            elif dim_name in ["service", "activity", "library", "provider", "receiver"]:
                dim_word_set["manifest"] = dim_word_set["manifest"].union(set(sample["manifest"]))
            elif dim_name in ["strings", "plurals", "string_arrays"]:
                dim_word_set["string"] = dim_word_set["string"].union(set(sample["string"]))
            else:
                dim_word_set[dim_name] = dim_word_set[dim_name].union(set(sample[dim_name]))
        if count == 1000:
            break

    # build feature vectors


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    feature_dir = os.path.abspath(config_json["feature_dir"])
    data_output_dir = os.path.abspath(config_json["data_output_dir"])
    model_output_dir = os.path.abspath(config_json["model_output_dir"])
    mode = config_json["mode"]

    if mode == "data":
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
                    cleaned_obj = clean_obj_text(feature_obj)
                    with open("%s/%s" % (data_output_dir, feature_path.split("/")[-1]), "w") as output_file:
                        json.dump(cleaned_obj, output_file)
    elif mode == "train":
        data_path_list = ["%s/%s" % (data_output_dir, x)
                          for x in os.walk(data_output_dir).next()[2]]
        print "assembling sample list..."
        # assemble sample list
        sample_list = []
        for data_path in data_path_list:
            with open(data_path, "r") as data_file:
                data_obj = json.load(data_file)
                if "class" in data_obj and len(data_obj["class"]) > 0:
                    sample_list.append(data_obj)
        print "%s samples read." % (len(sample_list))
        # build feature vectors
        print "building feature vectors..."
        build_word_bag(sample_list, config_json)

def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="train static features from apk")
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
