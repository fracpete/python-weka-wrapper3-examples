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

# dump_instances.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback

import weka.core.jvm as jvm
import weka.filters as filters
import wekaexamples.helper as helper
from simflow.control import Flow, run_flow
from simflow.source import FileSupplier
from weka.flow.sink import InstanceDumper
from weka.flow.transformer import LoadDataset, Filter, RenameRelation


def main():
    """
    Just runs some example code.
    """
    """
    Loads/filters a dataset incrementally and saves it to a new file.
    """

    # setup the flow
    helper.print_title("Load/filter/save dataset (incrementally)")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="Load/filter/save dataset (incrementally)")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    flow.actors.append(loaddataset)

    flter = Filter()
    flter.config["setup"] = filters.Filter(
        classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    flow.actors.append(flter)

    rename = RenameRelation()
    rename.config["name"] = "iris-reduced"
    flow.actors.append(rename)

    dumper = InstanceDumper()
    dumper.config["output"] = tempfile.gettempdir() + os.sep + "out.arff"
    flow.actors.append(dumper)

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
