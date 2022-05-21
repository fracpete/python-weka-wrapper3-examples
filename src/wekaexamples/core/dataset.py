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

# dataset.py
# Copyright (C) 2014-2022 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
from random import randint
import weka.core.jvm as jvm
from weka.core.classes import Random
import wekaexamples.helper as helper
from weka.core.converters import Loader
import weka.core.dataset as ds
from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.core.dataset import Attribute
import weka.plot.dataset as pld
import numpy as np


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    helper.print_title("Iris dataset")
    print(iris_data)
    helper.print_title("Iris dataset (incrementally output)")
    for i in iris_data:
        print(i)
    helper.print_title("Iris summary")
    print(Instances.summary(iris_data))
    helper.print_title("Iris attributes")
    for a in iris_data.attributes():
        print(a)
    helper.print_title("Instance at #0")
    print(iris_data.get_instance(0))
    print(iris_data.get_instance(0).values)
    print("Attribute stats (first):\n" + str(iris_data.attribute_stats(0)))
    print("total count (first attribute):\n" + str(iris_data.attribute_stats(0).total_count))
    print("numeric stats (first attribute):\n" + str(iris_data.attribute_stats(0).numeric_stats))
    print("nominal counts (last attribute):\n"
          + str(iris_data.attribute_stats(iris_data.num_attributes - 1).nominal_counts))
    helper.print_title("Instance values at #0")
    for v in iris_data.get_instance(0):
        print(v)

    # append datasets
    helper.print_title("append datasets")
    data1 = Instances.copy_instances(iris_data, 0, 2)
    data2 = Instances.copy_instances(iris_data, 2, 2)
    print("Dataset #1:\n" + str(data1))
    print("Dataset #2:\n" + str(data2))
    msg = data1.equal_headers(data2)
    print("#1 == #2 ? " + "yes" if msg is None else msg)
    combined = Instances.append_instances(data1, data2)
    print("Combined:\n" + str(combined))

    # merge datasets
    helper.print_title("merge datasets")
    data1 = Instances.copy_instances(iris_data, 0, 2)
    data1.class_index = -1
    data1.delete_attribute(1)
    data1.delete_first_attribute()
    data2 = Instances.copy_instances(iris_data, 0, 2)
    data2.class_index = -1
    data2.delete_attribute(4)
    data2.delete_attribute(3)
    data2.delete_attribute(2)
    print("Dataset #1:\n" + str(data1))
    print("Dataset #2:\n" + str(data2))
    msg = data1.equal_headers(data2)
    print("#1 == #2 ? " + ("yes" if msg is None else msg))
    combined = Instances.merge_instances(data2, data1)
    print("Combined:\n" + str(combined))

    # subsets
    helper.print_title("subsets")
    # select columns by name
    subset = iris_data.subset(col_names=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    print("col subset by name", subset.attribute_names(), subset.num_instances)

    # select columns by range (1-based indices)
    subset = iris_data.subset(col_range='1-3,5')
    print("col subset by range", subset.attribute_names(), subset.num_instances)

    # select rows by range (1-based indices)
    subset = iris_data.subset(row_range='51-150')
    print("row subset by range", subset.attribute_names(), subset.num_instances)

    # invert selection of cols/rows and keep original relation name
    subset = iris_data.subset(col_range='5', invert_cols=True, row_range='51-150', invert_rows=True, keep_relationame=True)
    print("col/row subset by range (inverted)", subset.attribute_names(), subset.num_instances)

    # load dataset incrementally
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset incrementally: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file, incremental=True)
    iris_data.class_is_last()
    helper.print_title("Iris dataset")
    print(iris_data)
    for inst in loader:
        print(inst)

    # create attributes
    helper.print_title("Creating attributes")
    num_att = Attribute.create_numeric("num")
    print("numeric: " + str(num_att))
    date_att = Attribute.create_date("dat", "yyyy-MM-dd")
    print("date: " + str(date_att))
    nom_att = Attribute.create_nominal("nom", ["label1", "label2"])
    print("nominal: " + str(nom_att))

    # create dataset
    helper.print_title("Create dataset")
    dataset = Instances.create_instances("helloworld", [num_att, date_att, nom_att], 0)
    print(str(dataset))

    # create an instance
    helper.print_title("Create and add instance")
    values = [3.1415926, date_att.parse_date("2014-04-10"), 1.0]
    inst = Instance.create_instance(values)
    print("Instance #1:\n" + str(inst))
    dataset.add_instance(inst)
    values = [2.71828, date_att.parse_date("2014-08-09"), Instance.missing_value()]
    inst = Instance.create_instance(values)
    dataset.add_instance(inst)
    print("Instance #2:\n" + str(inst))
    inst.set_value(0, 4.0)
    print("Instance #2 (updated):\n" + str(inst))
    print("Dataset:\n" + str(dataset))
    dataset.delete_with_missing(2)
    print("Dataset (after delete of missing):\n" + str(dataset))
    values = [(1, date_att.parse_date("2014-07-11"))]
    inst = Instance.create_sparse_instance(values, 3, classname="weka.core.SparseInstance")
    print("sparse Instance:\n" + str(inst))
    dataset.add_instance(inst)
    print("dataset with mixed dense/sparse instance objects:\n" + str(dataset))

    # create dataset (lists)
    helper.print_title("Create dataset from lists")
    x = [[randint(1, 10) for _ in range(5)] for _ in range(10)]
    y = [randint(0, 1) for _ in range(10)]
    dataset2 = ds.create_instances_from_lists(x, y, name="generated from lists")
    print(dataset2)
    x = [[randint(1, 10) for _ in range(5)] for _ in range(10)]
    dataset2 = ds.create_instances_from_lists(x, name="generated from lists (no y)")
    print(dataset2)

    # create dataset (mixed lists)
    helper.print_title("Create dataset from mixed lists")
    x = [["TEXT", 1, 1.1], ["XXX", 2, 2.2]]
    y = ["A", "B"]
    dataset2 = ds.create_instances_from_lists(x, y, name="generated from mixed lists", cols_x=["text", "integer", "float"], col_y="class")
    print(dataset2)
    dataset2 = ds.create_instances_from_lists(x, name="generated from mixed lists (no y)", cols_x=["text", "integer", "float"])
    print(dataset2)

    # create dataset (matrices)
    helper.print_title("Create dataset from matrices")
    x = np.random.randn(10, 5)
    y = np.random.randn(10)
    dataset3 = ds.create_instances_from_matrices(x, y, name="generated from matrices")
    print(dataset3)
    x = np.random.randn(10, 5)
    dataset3 = ds.create_instances_from_matrices(x, name="generated from matrix (no y)")
    print(dataset3)

    # create dataset (mixed type matrices)
    helper.print_title("Create dataset from (mixed type) matrix")
    x = np.array([("TEXT", 1, 1.1), ("XXX", 2, 2.2)], dtype='S20, i4, f8')
    y = np.array(["A", "B"], dtype='S20')
    dataset4 = ds.create_instances_from_matrices(x, y, name="generated from mixed matrices", cols_x=["text", "integer", "float"], col_y="class")
    print(dataset4)

    # create more sparse instances
    diabetes_file = helper.get_data_dir() + os.sep + "diabetes.arff"
    helper.print_info("Loading dataset: " + diabetes_file)
    loader = Loader("weka.core.converters.ArffLoader")
    diabetes_data = loader.load_file(diabetes_file)
    diabetes_data.class_is_last()
    helper.print_title("Create sparse instances using template dataset")
    sparse_data = Instances.template_instances(diabetes_data)
    for i in range(diabetes_data.num_attributes - 1):
        inst = Instance.create_sparse_instance(
            [(i, float(i+1) / 10.0)], sparse_data.num_attributes, classname="weka.core.SparseInstance")
        sparse_data.add_instance(inst)
    print("sparse dataset:\n" + str(sparse_data))

    # simple scatterplot of iris dataset: petalwidth x petallength
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    pld.scatter_plot(
        iris_data, iris_data.attribute_by_name("petalwidth").index,
        iris_data.attribute_by_name("petallength").index,
        percent=50,
        wait=False)

    # line plot of iris dataset (without class attribute)
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    pld.line_plot(iris_data, atts=range(iris_data.num_attributes - 1), percent=50, title="Line plot iris", wait=False)

    # matrix plot of iris dataset
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    pld.matrix_plot(iris_data, percent=50, title="Matrix plot iris", wait=True)

    # create cross-validation splits
    helper.print_title("Cross-validation splits")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    cv_splits = iris_data.cv_splits(3, rnd=Random(1), stratify=True)
    for i, cv_split in enumerate(cv_splits):
        helper.print_info("--> Split %d" % (i+1))
        print("\ntrain")
        print(cv_split[0])
        print("\ntest")
        print(cv_split[1])


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
