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

# learning_curve.py
# Copyright (C) 2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback
import weka.core.jvm as jvm
from weka.core.converters import load_any_file, save_any_file
from weka.filters import Filter, MultiFilter
from weka.classifiers import Classifier
from weka.experiments import SimpleCrossValidationExperiment, ResultMatrix, Tester
from weka.plot.experiments import plot_experiment
import wekaexamples.helper as helper


def main():
    """
    Just runs some example code.
    """

    # load data
    data_file = os.path.join(helper.get_data_dir(), "anneal.arff")
    print("Loading %s" % data_file)
    data = load_any_file(data_file, class_index="last")

    # generate datasets
    print("Generates learning curves using the experimenter")
    percentages = []
    filtered_datasets = dict()
    for p in range(10, 110, 20):
        percentages.append(p)
        remove = 100 - p
        print("dataset size: %d%%" % p)
        if remove > 0:
            randomize = Filter(classname="weka.filters.unsupervised.instance.Randomize", options=["-S", "1"])
            removeperc = Filter(classname="weka.filters.unsupervised.instance.RemovePercentage",
                                options=["-P", str(remove)])
            multi = MultiFilter()
            multi.filters = [randomize, removeperc]
            multi.inputformat(data)
            filtered = multi.filter(data)
            filtered.relationname = "%d%%" % p
            filtered_file = os.path.join(tempfile.gettempdir(), str(p) + ".arff")
            save_any_file(filtered, filtered_file)
            filtered_datasets[p] = filtered_file
        else:
            filtered_datasets[p] = data_file

    # setup experiment
    print("Configuring experiment")
    datasets = []
    for p in percentages:
        datasets.append(filtered_datasets[p])
    classifiers = [
        Classifier(classname="weka.classifiers.rules.ZeroR"),
        Classifier(classname="weka.classifiers.trees.J48"),
        Classifier(classname="weka.classifiers.trees.RandomForest")
    ]
    results_file = os.path.join(tempfile.gettempdir(), "results.arff")
    exp = SimpleCrossValidationExperiment(
        classification=True,
        runs=10,
        folds=10,
        datasets=datasets,
        classifiers=classifiers,
        result=results_file,
        pred_target_column=True,
        # outputting predictions and ground truth in separate columns (CAUTION: output can get very large!)
        class_for_ir_statistics=1)  # using 2nd class label for AUC

    # run experiment
    print("Running experiment")
    exp.setup()
    exp.run()

    # evaluate
    print("Evaluating experiment")
    results_data = load_any_file(results_file)
    matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText",
                          options=["-print-row-names", "-print-col-names"])
    # comparing datasets
    tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
    tester.swap_rows_and_cols = True
    tester.resultmatrix = matrix
    comparison_col = results_data.attribute_by_name("Percent_correct").index
    tester.instances = results_data
    print(tester.header(comparison_col))
    print(tester.multi_resultset_full(0, comparison_col))

    # plot
    print("Plotting results")
    plot_experiment(matrix, title="Learning curve", measure="Percent_correct",
                    key_loc="lower left", bbox_to_anchor=(0, 1, 1, 0),
                    y_label="Accuracy %", x_label="Dataset size: %s%%" % (",".join([str(x) for x in percentages])),
                    axes_swapped=True,
                    show_stdev=True, wait=True)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
