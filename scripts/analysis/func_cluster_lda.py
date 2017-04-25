import argparse
import json
import logging
import os
import pickle
import re
import sys

from multiprocessing import Process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    stream=sys.stdout)


def load_samples(config_json):
    feature_dir = os.path.abspath(config_json["feature_dir"])
    n_features = config_json["n_features"]
    word_combine = config_json["word_combine"]
    feature_path_list = ["%s/%s" % (feature_dir, x)
                         for x in os.walk(feature_dir).next()[2]]
    sample_list = {}
    count = 0
    for feature_path in feature_path_list:
        # logging.debug(feature_path)
        # logging.debug("%s/%s" % (str(count), str(len(feature_path_list))))
        count += 1
        with open(feature_path, "r") as feature_file:
            sample_obj = json.load(feature_file)
            for sample_key in sample_obj:
                if sample_key != "class":
                    if sample_key not in sample_list:
                        sample_list[sample_key] = []
                    sample_list[sample_key].append(" ".join(sample_obj[sample_key]))

    # confirm combine order
    sample_key_order = sorted(sample_list.keys())

    if word_combine:
        for idx in range(len(sample_list[sample_key_order[0]])):
            sample_list["permission"][idx] = sample_list["permission"][idx]
            sample_list["combined"][idx] = " ".join([sample_list[x][idx]
                                                     for x in sample_key_order if x != "permission"])
    # print sample_list.keys()

    ret_dict = {}
    for sample_key in sample_list:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           tokenizer=lambda x: x.split(),
                                           lowercase=False,
                                           max_features=n_features)
        tfidf = tfidf_vectorizer.fit_transform(sample_list[sample_key])
        ret_dict[sample_key] = {
            "matrix": tfidf,
            "vectorizer": tfidf_vectorizer
        }
        # print tfidf_vectorizer.get_feature_names()
    return ret_dict


def train_lda(config_json):
    print json.dumps(config_json, indent=2)
    loaded_samples = load_samples(config_json)

    n_topics = config_json["n_topics"]
    n_top_words = config_json["n_top_words"]
    max_iter = config_json["max_iter"]
    learning_offset = config_json["learning_offset"]

    train_result = {}

    for sample_key in loaded_samples:
        train_result[sample_key] = {}
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=max_iter,
                                        learning_method='online',
                                        learning_offset=learning_offset,
                                        random_state=0,
                                        verbose=0)
        lda.fit(loaded_samples[sample_key]["matrix"])
        feature_names = loaded_samples[sample_key]["vectorizer"].get_feature_names()
        for topic_idx, topic in enumerate(lda.components_):
            train_result[sample_key]["Topic #%d:" % topic_idx] = [feature_names[i]
                                                                  for i in topic.argsort()[:-n_top_words - 1:-1]]

    with open("%s/%s_%s_%s_%s.json" % (config_json["data_output_dir"],
              config_json["n_features"],
              config_json["n_topics"],
              config_json["max_iter"],
              config_json["learning_offset"]), "w") as output_file:
        json.dump(train_result, output_file, indent=2)


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


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))
    calc_process_set = set()

    for n_features in config_json["n_features"]:
        for n_topics in config_json["n_topics"]:
            for max_iter in config_json["max_iter"]:
                for learning_offset in config_json["learning_offset"]:
                    tmp_config_json = dict(config_json)
                    tmp_config_json["n_features"] = n_features
                    tmp_config_json["n_topics"] = n_topics
                    tmp_config_json["max_iter"] = max_iter
                    tmp_config_json["learning_offset"] = learning_offset
                    calc_process_set.add(Process(target=train_lda, args=[tmp_config_json]))
                    print n_features, n_topics, max_iter, learning_offset

    parallel_calc_task(calc_process_set, config_json["process_num"])


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="LDA clustering for static features from apk")
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
