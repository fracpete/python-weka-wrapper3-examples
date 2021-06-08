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

# silhouette_coefficient.py
# Copyright (C) 2021 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import load_any_file
from weka.clusterers import Clusterer, avg_silhouette_coefficient
from weka.core.distances import DistanceFunction
from weka.filters import Filter


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    helper.print_title("Loading and preparing iris dataset")
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    data = load_any_file(iris_file)
    data.delete_last_attribute()

    # filter dataset
    std = Filter(classname="weka.filters.unsupervised.attribute.Standardize")
    std.inputformat(data)
    data = std.filter(data)

    # computing average silhouette coefficient
    helper.print_title("Computing average silhouette coefficient")

    # Eculidean distance without normalization
    dist_func = DistanceFunction(classname="weka.core.EuclideanDistance", options=["-D"])

    clusterer = Clusterer("weka.clusterers.EM")
    clusterer.build_clusterer(data)
    print(clusterer.to_commandline() + "\n--> " + str(avg_silhouette_coefficient(clusterer, dist_func, data)))

    # we need to make sure that SimpleKMeans's distance function doesn't normalize
    clusterer = Clusterer("weka.clusterers.SimpleKMeans", options=["-N", "3", "-A", dist_func.to_commandline()])
    clusterer.build_clusterer(data)
    print(clusterer.to_commandline() + "\n--> " + str(avg_silhouette_coefficient(clusterer, dist_func, data)))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
