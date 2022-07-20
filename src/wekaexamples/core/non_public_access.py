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

# non_public_access.py
# Copyright (C) 2022 Fracpete (pythonwekawrapper at gmail dot com)

import javabridge
import os
import traceback

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.classes import new_instance, get_non_public_field, call_non_public_method, JavaObject
from weka.core.converters import load_any_file
from weka.core.distances import DistanceFunction
from weka.core.packages import is_installed, install_packages
from weka.core.typeconv import jdouble_matrix_to_ndarray
import wekaexamples.helper as helper


def main():
    """
    Just runs some example code.
    """

    # PrivateMember instance gets instantiated with a random integer
    helper.print_info("Accessing private field")
    pf = new_instance("weka.nonpublic.PrivateMember")
    # let's retrieve that random integer via the private method "value"
    value = get_non_public_field(pf, "value")
    jvalue = JavaObject(value)
    print(jvalue.jwrapper.intValue())

    # PrivateMethod instance gets instantiated with a random integer
    helper.print_info("Accessing private method")
    pm = new_instance("weka.nonpublic.PrivateMethod")
    # let's retrieve that random integer via the private method "getValue"
    value = call_non_public_method(pm, "getValue")
    jvalue = JavaObject(value)
    print(jvalue.jwrapper.intValue())
    # let's retrieve that random integer via the private method "getValueArg"
    # which also requires an int argument
    value = call_non_public_method(pm, "getValueArg", ["int"], [123])
    jvalue = JavaObject(value)
    print(jvalue.jwrapper.intValue())

    # access m_root of a built J48
    helper.print_info("Accessing private field of J48")
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    iris_data = load_any_file(iris_file, class_index="last")
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(iris_data)
    root = get_non_public_field(cls.jobject, "m_root")
    jroot = JavaObject(root)
    print(jroot.classname)
    print(jroot.jwrapper.field_names)
    print(jroot.jwrapper.methods)

    # Accessing protected matricesDilca field of the DilcaDistance class
    helper.print_info("Accessing private field of DilcaDistance")
    if not is_installed("DilcaDistance"):
        pkgs = ["fastCorrBasedFS", "DilcaDistance"]
        install_packages(pkgs)
        print("installed packages (%s), please rerun script" % str(pkgs))
        jvm.stop()
        import sys
        sys.exit(0)

    anneal_file = helper.get_data_dir() + os.sep + "anneal.arff"
    helper.print_info("Loading dataset: " + anneal_file)
    data = load_any_file(anneal_file, class_index="last")
    df = DistanceFunction(classname="weka.core.DilcaDistance")
    df.instances = data
    o = get_non_public_field(df.jobject, "matricesDilca")
    print(o)
    v = javabridge.get_collection_wrapper(o)
    for item in v:
        mat = jdouble_matrix_to_ndarray(item)
        print(mat)


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
