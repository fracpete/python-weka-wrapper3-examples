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

# build_evaluate_classifier.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback

import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, ContainerValuePicker, Trigger, run_flow
from simflow.sink import Console
from simflow.source import FileSupplier, Start, GetStorageValue
from weka.classifiers import Classifier
from weka.flow.transformer import LoadDataset, ClassSelector, Train, Evaluate, SetStorageValue, EvaluationSummary


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("build and evaluate classifier")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="build and evaluate classifier")

    start = Start()
    flow.actors.append(start)

    build_save = Trigger()
    build_save.name = "build and store classifier"
    flow.actors.append(build_save)

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    build_save.actors.append(filesupplier)

    loaddataset = LoadDataset()
    build_save.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    build_save.actors.append(select)

    ssv = SetStorageValue()
    ssv.config["storage_name"] = "data"
    build_save.actors.append(ssv)

    train = Train()
    train.config["setup"] = Classifier(classname="weka.classifiers.trees.J48")
    build_save.actors.append(train)

    pick = ContainerValuePicker()
    pick.config["value"] = "Model"
    build_save.actors.append(pick)

    ssv = SetStorageValue()
    ssv.config["storage_name"] = "model"
    pick.actors.append(ssv)

    evaluate = Trigger()
    evaluate.name = "evaluate classifier"
    flow.actors.append(evaluate)

    gsv = GetStorageValue()
    gsv.config["storage_name"] = "data"
    evaluate.actors.append(gsv)

    evl = Evaluate()
    evl.config["storage_name"] = "model"
    evaluate.actors.append(evl)

    summary = EvaluationSummary()
    summary.config["matrix"] = True
    evaluate.actors.append(summary)

    console = Console()
    evaluate.actors.append(console)

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
