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

# centroids.py
# Copyright (C) 2022-2024 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import load_any_file
from weka.core.dataset import Instances
from weka.clusterers import Clusterer


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)

    # delete last attribute
    data = load_any_file(iris_file)
    data.delete_attribute(data.num_attributes - 1)

    # build SimpleKMeans
    cls = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
    cls.build_clusterer(data)

    # jwrapper approach to get centroids
    print("jwrapper")
    centroids = cls.jwrapper.getClusterCentroids()
    for i in range(centroids.numInstances()):
        print(centroids.instance(i))

    # jni/pww approach to get centroids
    print("jni/pww")
    centroids = Instances(cls.jobject.getClusterCentroids())
    for i in range(centroids.num_instances):
        print(centroids.get_instance(i))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
