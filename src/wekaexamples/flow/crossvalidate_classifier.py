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

# crossvalidate_classifier.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback

import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, Branch, Sequence, run_flow
from simflow.sink import Console
from simflow.source import FileSupplier
from weka.classifiers import Classifier
from weka.flow.sink import ClassifierErrors, ROC, PRC
from weka.flow.transformer import LoadDataset, ClassSelector, CrossValidate, EvaluationSummary


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("Cross-validate classifier")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="cross-validate classifier")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    flow.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    flow.actors.append(select)

    cv = CrossValidate()
    cv.config["setup"] = Classifier(classname="weka.classifiers.trees.J48")
    flow.actors.append(cv)

    branch = Branch()
    flow.actors.append(branch)

    seqsum = Sequence()
    seqsum.name = "summary"
    branch.actors.append(seqsum)

    summary = EvaluationSummary()
    summary.config["title"] = "=== J48/iris ==="
    summary.config["complexity"] = False
    summary.config["matrix"] = True
    seqsum.actors.append(summary)

    console = Console()
    seqsum.actors.append(console)

    seqerr = Sequence()
    seqerr.name = "errors"
    branch.actors.append(seqerr)

    errors = ClassifierErrors()
    errors.config["wait"] = False
    seqerr.actors.append(errors)

    seqroc = Sequence()
    seqroc.name = "roc"
    branch.actors.append(seqroc)

    roc = ROC()
    roc.config["wait"] = False
    roc.config["class_index"] = [0, 1, 2]
    seqroc.actors.append(roc)

    seqprc = Sequence()
    seqprc.name = "prc"
    branch.actors.append(seqprc)

    prc = PRC()
    prc.config["wait"] = True
    prc.config["class_index"] = [0, 1, 2]
    seqprc.actors.append(prc)

    # run the flow
    run_flow(flow, print_tree=True, cleanup=True)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
