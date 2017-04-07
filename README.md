# Various tools for Android app analysis

## Dynamic Data Collection
* scripts/tester.py
    * parallely run DroidBot on multiple devices to dynamically test apps

## Static Data Collection
* scripts/static_feature_extractor
    * extract static text features from apps

## Analysis Tools
* scripts/data_cleaner.py
    * clean data collected from DroidBot, merge identical states, map method traces to corresponding transitions
* scripts/func_detector_train.py
    * clean text features collected by static_feature_extractor
    * select features based on mutual information
    * train, tune classifiers for app functionalities;
    * predict functionalities of an app
* visualizer/*
    * web pages for marking app functionalities manually


