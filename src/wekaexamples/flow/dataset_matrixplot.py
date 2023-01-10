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

# dataset_matrixplot.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback

import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, run_flow
from simflow.source import FileSupplier
from weka.flow.sink import MatrixPlot
from weka.flow.transformer import LoadDataset


def main():
    """
    Just runs some example code.
    """
    """
    Displays a dataset as matrixplot.
    """

    # setup the flow
    helper.print_title("Matrix plot")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="matrix plot")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    flow.actors.append(loaddataset)

    plot = MatrixPlot()
    plot.config["percent"] = 50.0
    flow.actors.append(plot)

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
