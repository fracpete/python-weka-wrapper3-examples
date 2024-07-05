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

# attribute_selection_test.py
# Copyright (C) 2014-2020 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.classifiers import Classifier, Evaluation, AttributeSelectedClassifier
import weka.filters as wfilters


def use_classifier(data):
    """
    Uses the meta-classifier AttributeSelectedClassifier for attribute selection.
    :param data: the dataset to use
    :type data: Instances
    """
    print("\n1. Meta-classifier")
    classifier = AttributeSelectedClassifier()
    aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    assearch = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-B"])
    base = Classifier(classname="weka.classifiers.trees.J48")
    # setting nested options is always a bit tricky, getting all the escaped double quotes right
    # simply using the bean property for setting Java objects is often easier and less error prone
    classifier.classifier = base
    classifier.evaluator = aseval
    classifier.search = assearch
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, 10, Random(1))
    print(evaluation.summary())
    print("Evaluator:\n", classifier.evaluator)
    print("Search:\n", classifier.search)


def use_filter(data):
    """
    Uses the AttributeSelection filter for attribute selection.
    :param data: the dataset to use
    :type data: Instances
    """
    print("\n2. Filter")
    flter = wfilters.AttributeSelection()
    aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    assearch = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-B"])
    flter.evaluator = aseval
    flter.search = assearch
    flter.inputformat(data)
    filtered = flter.filter(data)
    print(str(filtered))
    print("Evaluator:\n", flter.evaluator)
    print("Search:\n", flter.search)


def use_low_level(data):
    """
    Uses the attribute selection API directly.
    :param data: the dataset to use
    :type data: Instances
    """
    print("\n3. Low-level")
    attsel = AttributeSelection()
    aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    assearch = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-B"])
    attsel.jobject.setEvaluator(aseval.jobject)
    attsel.jobject.setSearch(assearch.jobject)
    attsel.select_attributes(data)
    indices = attsel.selected_attributes
    print("selected attribute indices (starting with 0):\n" + str(indices.tolist()))


def main(args):
    """
    Performs attribute selection on the specified dataset (uses vote UCI dataset if no dataset specified). Last
    attribute is assumed to be the class attribute. Used: CfsSubsetEval, GreedyStepwise, J48
    :param args: the commandline arguments
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        data_file = helper.get_data_dir() + os.sep + "vote.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    use_classifier(data)
    use_filter(data)
    use_low_level(data)

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
