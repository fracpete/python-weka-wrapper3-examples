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

# bayesnet.py
# Copyright (C) 2022 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import from_commandline
from weka.core.converters import load_any_file
from weka.plot.graph import plot_xmlbif_graph, xmlbif_to_dot


def main():
    """
    Shows how to plot the belief network generated by BayesNet.
    """

    # load a dataset
    data_file = helper.get_data_dir() + os.sep + "glass.arff"
    helper.print_info("Loading dataset: " + data_file)
    data = load_any_file(data_file, class_index="last")

    # classifier
    classifier = from_commandline("weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 2 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5",
                                  classname="weka.classifiers.Classifier")
    classifier.build_classifier(data)
    xmlbif = classifier.graph
    print("\nXML BIF\n\n", xmlbif)
    dot = xmlbif_to_dot(xmlbif)
    print("\nDOT\n\n", dot)
    plot_xmlbif_graph(xmlbif)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()