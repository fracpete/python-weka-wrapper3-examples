# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# output_class_distribution.py
# Copyright (C) 2014-2021 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main(args):
    """
    Trains a J48 classifier on a training set and outputs the predicted class and class distribution alongside the
    actual class from a test set. Class attribute is assumed to be the last attribute.
    :param args: the commandline arguments (train and test datasets, optional)
    :type args: list
    """

    # load a dataset
    if len(args) < 3:
        helper.print_info("No custom dataset filenames provided, falling back on default ones")
        train_file = helper.get_data_dir() + os.sep + "iris-train.arff"
        test_file = helper.get_data_dir() + os.sep + "iris-test.arff"
    else:
        train_file = args[1]
        test_file = args[2]
    helper.print_info("Loading train: " + train_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(train_file)
    train.class_index = train.num_attributes - 1
    helper.print_info("Loading test: " + test_file)
    test = loader.load_file(test_file)
    test.class_is_last()

    # classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)

    # output predictions
    print("# - actual - predicted - error - distribution")
    for index, inst in enumerate(test):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print(
            "%d - %s - %s - %s  - %s" %
            (index+1,
             inst.get_string_value(inst.class_index),
             inst.class_attribute.value(int(pred)),
             "yes" if pred != inst.get_value(inst.class_index) else "no",
             str(dist.tolist())))


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
