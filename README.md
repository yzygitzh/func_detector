# Various tools for Android app analysis

## Dynamic Data Collection
* scripts/dynamic/tester.py
    * parallely run DroidBot on multiple devices to dynamically test apps
* scripts/dynamic/droidbot_data_cleaner.py
    * clean data collected from DroidBot, merge identical states, map method traces to corresponding transitions

## Static Data Collection
* scripts/static/static_feature_extractor.py
    * extract static text features from apps
    * apps must be named after their package names
    * app package name lists named in top_apps_in_SOMECLASS.txt must be provided
* scripts/static/static_feature_cleaner.py
    * clean the features extracted from `static_feature_extractor.py`
    * repeat/combine options are available
        * repeat: allow for repeat words
        * combine: combine different feature keys into `combined`

## Analysis Tools
* scripts/analysis/func_detector_train.py
    * clean text features collected by static_feature_extractor (into sets, without tf info)
    * select features based on mutual information
    * train, tune classifiers for app functionalities
    * predict functionalities of an app
* scripts/analysis/func_cluster_lda.py
    * train LDA clusters for app functionalities
* visualizer/*
    * web pages for marking app functionalities manually


