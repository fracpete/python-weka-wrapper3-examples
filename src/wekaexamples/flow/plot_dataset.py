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

# plot_dataset.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback

import weka.core.jvm as jvm
import weka.filters as filters
import wekaexamples.helper as helper
from simflow.control import Flow, Branch, Sequence, run_flow
from simflow.source import FileSupplier
from weka.flow.sink import MatrixPlot, LinePlot
from weka.flow.transformer import LoadDataset, Filter, Copy


def main():
    """
    Just runs some example code.
    """
    """
    Plots a dataset.
    """

    # setup the flow
    helper.print_title("Plot dataset")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="plot dataset")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    flow.actors.append(loaddataset)

    branch = Branch()
    flow.actors.append(branch)

    seq = Sequence(name="matrix plot")
    branch.actors.append(seq)

    mplot = MatrixPlot()
    mplot.config["percent"] = 50.0
    mplot.config["wait"] = False
    seq.actors.append(mplot)

    seq = Sequence(name="line plot")
    branch.actors.append(seq)

    copy = Copy()
    seq.actors.append(copy)

    flter = Filter()
    flter.config["setup"] = filters.Filter(
        classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    flter.config["keep_relationname"] = True
    seq.actors.append(flter)

    lplot = LinePlot()
    lplot.config["percent"] = 50.0
    lplot.config["wait"] = True
    seq.actors.append(lplot)

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
