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

# build_classifier_incrementally.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback

import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, ContainerValuePicker, Tee, Trigger, run_flow
from simflow.sink import Console
from simflow.source import FileSupplier, GetStorageValue
from simflow.transformer import InitStorageValue, UpdateStorageValue
from weka.classifiers import Classifier
from weka.flow.transformer import LoadDataset, ClassSelector, Train


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    count = 50
    helper.print_title("build classifier incrementally")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="build classifier incrementally")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    initcounter = InitStorageValue()
    initcounter.config["storage_name"] = "counter"
    initcounter.config["value"] = 0
    flow.actors.append(initcounter)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    flow.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    flow.actors.append(select)

    inccounter = UpdateStorageValue()
    inccounter.config["storage_name"] = "counter"
    inccounter.config["expression"] = "{X} + 1"
    flow.actors.append(inccounter)

    train = Train()
    train.config["setup"] = Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")
    flow.actors.append(train)

    pick = ContainerValuePicker()
    pick.config["value"] = "Model"
    pick.config["switch"] = True
    flow.actors.append(pick)

    tee = Tee(name="output model every " + str(count) + " instances")
    tee.config["condition"] = "@{counter} % " + str(count) + " == 0"
    flow.actors.append(tee)

    trigger = Trigger(name="output # of instances")
    tee.actors.append(trigger)

    getcounter = GetStorageValue()
    getcounter.config["storage_name"] = "counter"
    trigger.actors.append(getcounter)

    console = Console()
    console.config["prefix"] = "# of instances: "
    trigger.actors.append(console)

    console = Console(name="output model")
    tee.actors.append(console)

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
