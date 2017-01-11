from multiprocessing import Process

import json
import os
import argparse
import re
import logging

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_selection import SelectKBest, mutual_info_classif
import scipy.sparse

# some init
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = stopwords.words("english")
first_cap_re = re.compile("(.)([A-Z][a-z]+)")
all_cap_re = re.compile("([a-z0-9])([A-Z])")

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    filename="func_detector_train.log",
                    filemode="w")


def id_convert(name):
    s1 = first_cap_re.sub(r"\1_\2", name)
    return all_cap_re.sub(r"\1_\2", s1).lower()


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
            logging.warning(e)
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

    for feature_key in ret_obj:
        ret_obj[feature_key] = list(ret_obj[feature_key])

    return ret_obj


def cast_sample(sample, vec_space):
    # cast_sample["manifest"] = [0, 0, 1, ...]
    ret_list = []
    for word in vec_space:
        if word in sample:
            ret_list.append(1)
        else:
            ret_list.append(0)
    return ret_list


def mi_worker(X, y, label, dim_name, dim_word_list, output_path):
    logging.debug("[MI START]: %s, %s" % (label, dim_name))
    mi = mutual_info_classif(X, y, discrete_features=True)
    logging.debug("[MI END]: %s, %s" % (label, dim_name))
    sorted_mi = sorted(enumerate(mi), key=lambda x:x[1], reverse=True)
    feature_text_list = [(dim_word_list[dim_name][x[0]], x[1]) for x in sorted_mi]
    with open("%s/%s-%s.json" % (output_path, label, dim_name), "w") as output_file:
        json.dump(feature_text_list, output_file, indent=2)


def select_feature(sample_list, config_json):
    # select features in different dimensions by calculating mi
    # using sklearn's method

    # calc label set
    # calc the whole word set for feature dim"s
    label_set = set()
    dim_word_set = {
        "manifest": set(),
        "string": set(),
        "public": set(),
        "permission": set()
    }
    count = 0
    for sample in sample_list:
        logging.debug("[Counting Words]: %s/%s" % (str(count), str(len(sample_list))))
        label_set = label_set.union(set(sample["class"]))
        for dim_name in ["service", "activity", "library", "provider", "receiver"]:
            dim_word_set["manifest"] = dim_word_set["manifest"].union(set(sample[dim_name]))
        for dim_name in ["strings", "plurals", "string_arrays"]:
            dim_word_set["string"] = dim_word_set["string"].union(set(sample[dim_name]))
        for dim_name in ["permission", "public"]:
            dim_word_set[dim_name] = dim_word_set[dim_name].union(set(sample[dim_name]))
        count += 1

    # build list for mi calc, transform back to set later
    dim_word_list = {}
    for dim_name in dim_word_set:
        dim_word_list[dim_name] = list(dim_word_set[dim_name])

    # build sample vecs
    # sample_vec["WEATHER"][0]["manifest"] = ["weather", "rain", ...]
    sample_vec = {}
    for label in label_set:
        sample_vec[label] = []

    count = 0
    for sample in sample_list:
        logging.debug("[Building Samples]: %s/%s" % (str(count), str(len(sample_list))))
        vec = {
            "manifest": set(),
            "string": set(),
            "public": set(),
            "permission": set()
        }
        for dim_name in ["service", "activity", "library", "provider", "receiver"]:
            vec["manifest"] = vec["manifest"].union(set(sample[dim_name]))
        for dim_name in ["strings", "plurals", "string_arrays"]:
            vec["string"] = vec["string"].union(set(sample[dim_name]))
        for dim_name in ["permission", "public"]:
            vec[dim_name] = vec[dim_name].union(set(sample[dim_name]))

        for dim_name in vec:
            vec[dim_name] = scipy.sparse.csr_matrix(
                cast_sample(vec[dim_name], dim_word_list[dim_name]))

        for label in sample["class"]:
            sample_vec[label].append(vec)
        count += 1

    mi_calc_task_pool = set()
    for label in label_set:
        for dim_name in dim_word_set:
            normalized_positive_sample_list = [x[dim_name] for x in sample_vec[label]]
            positive_label_list = [1 for x in range(len(normalized_positive_sample_list))]

            normalized_negative_sample_list = []
            for other_label in label_set - set([label]):
                normalized_negative_sample_list += [x[dim_name] for x in sample_vec[other_label]]
            negative_label_list = [0 for x in range(len(normalized_negative_sample_list))]

            X = normalized_positive_sample_list + normalized_negative_sample_list
            X = scipy.sparse.vstack(X, format="csr")
            y = positive_label_list + negative_label_list

            mi_calc_task_pool.add(Process(target=mi_worker, args=(
                X, y, label, dim_name, dim_word_list, config_json["selected_feature_output_dir"])))

    while len(mi_calc_task_pool) > 0:
        task_batch = []
        for i in range(config_json["process_num"]):
            if len(mi_calc_task_pool) == 0:
                break
            task_batch.append(mi_calc_task_pool.pop())
            task_batch[-1].start()
        for task in task_batch:
            task.join()


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    feature_dir = os.path.abspath(config_json["feature_dir"])
    data_output_dir = os.path.abspath(config_json["data_output_dir"])
    model_output_dir = os.path.abspath(config_json["model_output_dir"])
    mode = config_json["mode"]

    if mode == "clean_text":
        feature_path_list = ["%s/%s" % (feature_dir, x)
                             for x in os.walk(feature_dir).next()[2]]
        count = 0
        for feature_path in feature_path_list:
            logging.debug(feature_path)
            logging.debug("%s/%s" % (str(count), str(len(feature_path_list))))
            count += 1
            with open(feature_path, "r") as feature_file:
                feature_obj = json.load(feature_file)
                if "class" in feature_obj and len(feature_obj["class"]) > 0:
                    cleaned_obj = clean_obj_text(feature_obj)
                    with open("%s/%s" % (data_output_dir, feature_path.split("/")[-1]), "w") as output_file:
                        json.dump(cleaned_obj, output_file, indent=2)
    elif mode == "select_feature":
        data_path_list = ["%s/%s" % (data_output_dir, x)
                          for x in os.walk(data_output_dir).next()[2]]
        logging.debug("assembling sample list...")
        # assemble sample list
        sample_list = []
        for data_path in data_path_list:
            with open(data_path, "r") as data_file:
                data_obj = json.load(data_file)
                if "class" in data_obj and len(data_obj["class"]) > 0:
                    sample_list.append(data_obj)
        logging.debug("%s samples read." % (len(sample_list)))
        # build feature vectors
        logging.debug("building feature vectors...")
        select_feature(sample_list, config_json)

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
