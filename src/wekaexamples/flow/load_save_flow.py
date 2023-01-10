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

# load_save_flow.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback

import simflow.conversion as conversion
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, Tee, run_flow
from simflow.sink import Console
from simflow.source import ListFiles
from simflow.transformer import Convert
from weka.classifiers import Classifier
from weka.flow.transformer import LoadDataset, CrossValidate, EvaluationSummary


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="list files")

    listfiles = ListFiles()
    listfiles.config["dir"] = str(helper.get_data_dir())
    listfiles.config["list_files"] = True
    listfiles.config["list_dirs"] = False
    listfiles.config["recursive"] = False
    listfiles.config["regexp"] = ".*.arff"
    flow.actors.append(listfiles)

    tee = Tee()
    flow.actors.append(tee)

    convert = Convert()
    convert.config["setup"] = conversion.PassThrough()
    tee.actors.append(convert)

    console = Console()
    console.config["prefix"] = "Match: "
    tee.actors.append(console)

    load = LoadDataset()
    load.config["use_custom_loader"] = True
    flow.actors.append(load)

    cross = CrossValidate()
    cross.config["setup"] = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
    flow.actors.append(cross)

    summary = EvaluationSummary()
    summary.config["matrix"] = True
    flow.actors.append(summary)

    # print flow
    flow.setup()
    print("\n" + flow.tree + "\n")

    # save the flow
    fname = tempfile.gettempdir() + os.sep + "simpleflow.json"
    Flow.save(flow, fname)

    # load flow
    fl2 = Flow.load(fname)

    # output flow
    fl2.setup()
    print("\n" + fl2.tree + "\n")


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
