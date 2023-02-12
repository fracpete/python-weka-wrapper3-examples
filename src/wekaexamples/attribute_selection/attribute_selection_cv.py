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

# attribute_selection_cv.py
# Copyright (C) 2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import from_commandline, get_classname
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    anneal_file = helper.get_data_dir() + os.sep + "anneal.arff"
    helper.print_info("Loading dataset: " + anneal_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    anneal_data = loader.load_file(anneal_file)
    anneal_data.class_is_last()

    # instantiate search/evaluation from commandlines
    helper.print_title("Attribute selection (cross-validation - subset evaluation)")
    search = from_commandline('weka.attributeSelection.GreedyStepwise -B -T -1.7976931348623157E308 -N -1 -num-slots 1', classname=get_classname(ASSearch))
    print("search:", search.to_commandline())
    evaluation = from_commandline('weka.attributeSelection.CfsSubsetEval -P 1 -E 1', classname=get_classname(ASEvaluation))
    print("evaluation:", evaluation.to_commandline())

    attsel = AttributeSelection()
    attsel.crossvalidation(True)
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    print("\nsubset string:" + attsel.cv_results)
    print("subset list:\n" + str(attsel.subset_results))

    # instantiate search/evaluation from commandlines
    helper.print_title("Attribute selection (cross-validation - ranking)")
    search = from_commandline('weka.attributeSelection.Ranker', classname=get_classname(ASSearch))
    print("search:", search.to_commandline())
    evaluation = from_commandline('weka.attributeSelection.InfoGainAttributeEval', classname=get_classname(ASEvaluation))
    print("evaluation:", evaluation.to_commandline())

    attsel = AttributeSelection()
    attsel.crossvalidation(True)
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    res = attsel.cv_results
    print("\nrank string:" + res)
    print("rank dictionary:\n" + str(attsel.rank_results))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
