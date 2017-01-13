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
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

import scipy.sparse
import pickle

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


def parallel_calc_task(calc_process_set, process_num):
    while len(calc_process_set) > 0:
        task_batch = []
        for i in range(process_num):
            if len(calc_process_set) == 0:
                break
            task_batch.append(calc_process_set.pop())
            task_batch[-1].start()
        for task in task_batch:
            task.join()


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
    ret_list = []
    for word in vec_space:
        if word in sample:
            ret_list.append(1)
        else:
            ret_list.append(0)
    return ret_list


def clean_sample(sample, dim_word_list):
    # activity, receiver, ... => manifest, ...
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
    return vec


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
        vec = clean_sample(sample, dim_word_list)
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

    parallel_calc_task(mi_calc_task_pool, config_json["process_num"])


def trainer(sample_vec, model_output_path, dim_num_label):
    # sample_vec["WEATHER"][sample01, sample02, ...]
    label_set = set(sample_vec.keys())
    score_dict = {}
    for label in label_set:
        # assemble sample list
        logging.debug("[Assemble Class]: %s" % str(label))
        positive_list = sample_vec[label]
        negative_list = []
        for other_label in (label_set - set([label])):
            negative_list.extend(sample_vec[other_label])
        X = scipy.sparse.vstack(positive_list + negative_list)
        y = [1 for x in positive_list] + [0 for x in negative_list]
        X, y = shuffle(X, y)
        clf = svm.SVC(class_weight="balanced")
        logging.debug("[CV Training %s]: %s" % (str(dim_num_label), str(label)))
        scores_precision = list(cross_val_score(clf, X, y, cv=5, scoring="precision"))
        logging.debug("[Precision %s]: %s: %s" % (str(dim_num_label), str(label), str(scores_precision)))
        scores_recall = list(cross_val_score(clf, X, y, cv=5, scoring="recall"))
        logging.debug("[Recall %s]: %s: %s" % (str(dim_num_label), str(label), str(scores_recall)))

        score_dict[label] = {"precision": scores_precision, "recall": scores_recall}
        # logging.debug("[Full Training]: %s" % str(label))
        # clf.fit(X, y)
        # with open("%s/%s.model" % (model_output_path, label), "w") as model_file:
        #    pickle.dump(clf, model_file)
    with open("%s/%s.train_result" % (model_output_path, dim_num_label), "w") as train_result_file:
        json.dump(score_dict, train_result_file, indent=2)


def predicter(named_cleaned_sample_list, label, clf, predict_output_dir):
    logging.debug("[Predicting]: %s" % str(label))
    X = scipy.sparse.vstack([x[1] for x in named_cleaned_sample_list])
    sample_name_list = [x[0] for x in named_cleaned_sample_list]
    y_pred = clf.predict(X)
    predict_result = [[sample_name_list[i], y_pred[i]] for i in range(len(sample_name_list))]
    with open("%s/%s.predict" % (predict_output_dir, label), "w") as predict_file:
        json.dump(predict_result, predict_file, indent=2)


def read_samples(data_output_dir, predict_mode=False):
    data_path_list = ["%s/%s" % (data_output_dir, x)
                      for x in os.walk(data_output_dir).next()[2]]
    # assemble sample list
    sample_list = []
    count = 0
    for data_path in data_path_list:
        # logging.debug("[Reading Samples]: %s/%s" % (str(count), str(len(data_path_list))))
        with open(data_path, "r") as data_file:
            data_obj = json.load(data_file)
            if predict_mode:
                sample_list.append((data_path, data_obj))
            elif "class" in data_obj and len(data_obj["class"]) > 0:
                sample_list.append(data_obj)
        count += 1
    return sample_list


def read_features(selected_feature_output_dir):
    # feature_dict["WEATHER"]["manifest"]
    selected_feature_list = [(x, "%s/%s" % (selected_feature_output_dir, x))
                             for x in os.walk(selected_feature_output_dir).next()[2]]
    feature_dict = {}
    count = 0
    for feature_tuple in selected_feature_list:
        # logging.debug("[Reading Features]: %s/%s" % (str(count), str(len(selected_feature_list))))
        class_name, dim_name = feature_tuple[0][:-len(".json")].split("-")
        if class_name not in feature_dict:
            feature_dict[class_name] = {}
        with open(feature_tuple[1], "r") as feature_file:
            feature_list = json.load(feature_file)
            feature_dict[class_name][dim_name] = [x[0] for x in feature_list]
        count += 1
    return feature_dict


def clean_features(feature_dict, feature_dim_number):
    ret_dict = {}
    for class_name in feature_dict:
        ret_dict[class_name] = {}
        for dim_name in feature_dict[class_name]:
            ret_dict[class_name][dim_name] = feature_dict[class_name][dim_name][:feature_dim_number[dim_name]]
    return ret_dict


def read_models(model_output_dir):
    model_list = [(x, "%s/%s" % (model_output_dir, x))
                  for x in os.walk(model_output_dir).next()[2]]
    model_dict = {}
    count = 0
    for model_tuple in model_list:
        logging.debug("[Reading Model]: %s/%s" % (str(count), str(len(model_list))))
        with open(model_tuple[1], "r") as model_file:
            clf = pickle.load(model_file)
            model_dict[model_tuple[0][:-len(".model")]] = clf
        count += 1
    return model_dict


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))

    feature_dir = os.path.abspath(config_json["feature_dir"])
    data_output_dir = os.path.abspath(config_json["data_output_dir"])
    selected_feature_output_dir = os.path.abspath(config_json["selected_feature_output_dir"])
    model_output_dir = os.path.abspath(config_json["model_output_dir"])
    mode = config_json["mode"]
    feature_dim_number = config_json["feature_dim_number"]

    logging.debug("MODE %s" % mode)

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
        # assemble sample list
        sample_list = read_samples(data_output_dir)
        # build feature vectors
        logging.debug("building feature vectors...")
        select_feature(sample_list, config_json)
    elif mode in ["training", "predicting"]:
        dim_number_combinations = [{"manifest": a, "string": b, "public": c, "permission": d}
                                   for a in feature_dim_number["manifest"]
                                   for b in feature_dim_number["string"]
                                   for c in feature_dim_number["public"]
                                   for d in feature_dim_number["permission"]]
        calc_process_set = set()
        # read back features
        origin_feature_dict = read_features(selected_feature_output_dir)
        for feature_dim_number_item in dim_number_combinations:
            logging.debug(str(feature_dim_number_item))
            # clean feature
            feature_dict = clean_features(origin_feature_dict, feature_dim_number_item)
            dim_number_label = "_".join([str(x[1]) for x in
                                         sorted(feature_dim_number_item.items(), key=lambda x: x[0])])
            # assemble sample list, do cleaning
            if mode == "training":
                sample_list = read_samples(data_output_dir)
                sample_vec = {}
                count = 0
                for sample in sample_list:
                    # logging.debug("[Cleaning Samples]: %s/%s" % (str(count), str(len(sample_list))))
                    for label in sample["class"]:
                        cleaned_sample = clean_sample(sample, feature_dict[label])
                        cleaned_sample = scipy.sparse.hstack([cleaned_sample[dim_name]
                                                            for dim_name in sorted(cleaned_sample.keys())])
                        if label not in sample_vec:
                            sample_vec[label] = []
                        sample_vec[label].append(cleaned_sample)
                    count += 1
                calc_process_set.add(Process(target=trainer, args=(
                    sample_vec, config_json["model_output_dir"], dim_number_label)))
            else:
                # read back model
                model_dict = read_models(model_output_dir)
                # read samples
                named_sample_list = read_samples(data_output_dir, predict_mode=True)
                for label in model_dict:
                    named_cleaned_sample_list = []
                    for named_sample in named_sample_list:
                        cleaned_sample = clean_sample(named_sample[1], feature_dict[label])
                        cleaned_sample = scipy.sparse.hstack([cleaned_sample[dim_name]
                                                            for dim_name in sorted(cleaned_sample.keys())])
                        named_cleaned_sample_list.append((named_sample[0], cleaned_sample))
                    predicter(named_cleaned_sample_list, label, model_dict[label], config_json["predict_output_dir"])

        parallel_calc_task(calc_process_set, config_json["process_num"])


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
