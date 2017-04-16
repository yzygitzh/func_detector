import argparse
import json
import logging
import os
import pickle
import re
import sys

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
        logging.debug(feature_path)
        logging.debug("%s/%s" % (str(count), str(len(feature_path_list))))
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
    print sample_list.keys()

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
        print tfidf_vectorizer.get_feature_names()
    return ret_dict


def train_lda(config_json, loaded_samples):
    n_topics = config_json["n_topics"]
    n_top_words = config_json["n_top_words"]
    for sample_key in loaded_samples:
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                        learning_method='online',
                                        learning_offset=1.,
                                        random_state=0,
                                        verbose=2)
        lda.fit(loaded_samples[sample_key]["matrix"])
        feature_names = loaded_samples[sample_key]["vectorizer"].get_feature_names()
        for topic_idx, topic in enumerate(lda.components_):
            print "Topic #%d:" % topic_idx
            print " ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])


def run(config_json_path):
    """
    parse config file and assign work to multiple processes
    """
    config_json = json.load(open(os.path.abspath(config_json_path), "r"))
    loaded_samples = load_samples(config_json)
    train_lda(config_json, loaded_samples)



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
