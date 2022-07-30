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

# capabilities.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.capabilities import Capability
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.core.capabilities import Capabilities


def main():
    """
    Just runs some example code.
    """

    classifier = Classifier(classname="weka.classifiers.trees.J48")

    helper.print_title("Capabilities")
    capabilities = classifier.capabilities
    print(capabilities)

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    data_capabilities = Capabilities.for_instances(iris_data)
    print(data_capabilities)
    print("classifier handles dataset: " + str(capabilities.supports(data_capabilities)))

    # disable/enable
    helper.print_title("Disable/Enable")
    capability = Capability(member="UNARY_ATTRIBUTES")
    capabilities.disable(capability)
    capabilities.min_instances = 10
    print("Removing: " + str(capability))
    print(capabilities)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
