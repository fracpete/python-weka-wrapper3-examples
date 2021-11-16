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

# experiments.py
# Copyright (C) 2014-2021 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback
import weka.core.jvm as jvm
import weka.core.converters as converters
import wekaexamples.helper as helper
from weka.classifiers import Classifier
from weka.experiments import SimpleCrossValidationExperiment, SimpleRandomSplitExperiment, Tester, ResultMatrix
import weka.plot.experiments as plot_exp


def main():
    """
    Just runs some example code.
    """

    print(helper.get_data_dir())

    # cross-validation + classification
    helper.print_title("Experiment: Cross-validation + classification")
    datasets = [helper.get_data_dir() + os.sep + "iris.arff", helper.get_data_dir() + os.sep + "anneal.arff"]
    classifiers = [Classifier("weka.classifiers.rules.ZeroR"), Classifier("weka.classifiers.trees.J48")]
    outfile = tempfile.gettempdir() + os.sep + "results-cv.arff"
    exp = SimpleCrossValidationExperiment(
        classification=True,
        runs=10,
        folds=10,
        datasets=datasets,
        classifiers=classifiers,
        result=outfile)
    exp.setup()
    exp.run()

    # evaluate
    loader = converters.loader_for_file(outfile)
    data = loader.load_file(outfile)
    matrix = ResultMatrix("weka.experiment.ResultMatrixPlainText")
    tester = Tester("weka.experiment.PairedCorrectedTTester")
    tester.resultmatrix = matrix
    comparison_col = data.attribute_by_name("Percent_correct").index
    tester.instances = data
    print(tester.header(comparison_col))
    print(tester.multi_resultset_full(0, comparison_col))

    # random split + regression
    helper.print_title("Experiment: Random split + regression")
    datasets = [helper.get_data_dir() + os.sep + "bolts.arff", helper.get_data_dir() + os.sep + "bodyfat.arff"]
    classifiers = [
        Classifier("weka.classifiers.rules.ZeroR"),
        Classifier("weka.classifiers.functions.LinearRegression")
    ]
    outfile = tempfile.gettempdir() + os.sep + "results-rs.arff"
    exp = SimpleRandomSplitExperiment(
        classification=False,
        runs=10,
        percentage=66.6,
        preserve_order=False,
        datasets=datasets,
        classifiers=classifiers,
        result=outfile)
    exp.setup()
    exp.run()

    # evaluate
    loader = converters.loader_for_file(outfile)
    data = loader.load_file(outfile)
    matrix = ResultMatrix("weka.experiment.ResultMatrixPlainText")
    tester = Tester("weka.experiment.PairedCorrectedTTester")
    tester.resultmatrix = matrix
    comparison_col = data.attribute_by_name("Correlation_coefficient").index
    tester.instances = data
    print(tester.header(comparison_col))
    print(tester.multi_resultset_full(0, comparison_col))

    # plot
    plot_exp.plot_experiment(matrix, title="Random split", measure="Correlation coefficient",
                             show_stdev=True, wait=True)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
