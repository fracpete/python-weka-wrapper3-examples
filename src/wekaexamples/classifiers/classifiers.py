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

# classifiers.py
# Copyright (C) 2014-2019 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier
from weka.classifiers import Evaluation
from weka.filters import Filter
from weka.core.classes import Random, from_commandline
import weka.plot.classifiers as plot_cls
import weka.plot.graph as plot_graph
import weka.core.typeconv as typeconv


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()

    # classifier help
    helper.print_title("Creating help string")
    classifier = Classifier(classname="weka.classifiers.trees.J48")
    print(classifier.to_help())

    # partial classname
    helper.print_title("Creating classifier from partial classname")
    clsname = ".J48"
    classifier = Classifier(classname=clsname)
    print(clsname + " --> " + classifier.classname)

    # classifier from commandline
    helper.print_title("Creating SMO from command-line string")
    cmdline = 'weka.classifiers.functions.SMO -K "weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 3.0"'
    classifier = from_commandline(cmdline, classname="weka.classifiers.Classifier")
    classifier.build_classifier(iris_data)
    print("input: " + cmdline)
    print("output: " + classifier.to_commandline())
    print("model:\n" + str(classifier))

    # kernel classifier
    helper.print_title("Creating SMO as KernelClassifier")
    kernel = Kernel(classname="weka.classifiers.functions.supportVector.RBFKernel", options=["-G", "0.001"])
    classifier = KernelClassifier(classname="weka.classifiers.functions.SMO", options=["-M"])
    classifier.kernel = kernel
    classifier.build_classifier(iris_data)
    print("classifier: " + classifier.to_commandline())
    print("model:\n" + str(classifier))

    # build a classifier and output model
    helper.print_title("Training J48 classifier on iris")
    classifier = Classifier(classname="weka.classifiers.trees.J48")
    # Instead of using 'options=["-C", "0.3"]' in the constructor, we can also set the "confidenceFactor"
    # property of the J48 classifier itself. However, being of type float rather than double, we need
    # to convert it to the correct type first using the double_to_float function:
    classifier.set_property("confidenceFactor", typeconv.float_to_jfloat(0.3))
    classifier.build_classifier(iris_data)
    print(classifier)
    print(classifier.graph)
    print(classifier.to_source("MyJ48"))
    plot_graph.plot_dot_graph(classifier.graph)

    # evaluate model on test set
    helper.print_title("Evaluating J48 classifier on iris")
    evaluation = Evaluation(iris_data)
    evl = evaluation.test_model(classifier, iris_data)
    print(evl)
    print(evaluation.summary())

    # evaluate model on train/test split
    helper.print_title("Evaluating J48 classifier on iris (random split 66%)")
    classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
    evaluation = Evaluation(iris_data)
    evaluation.evaluate_train_test_split(classifier, iris_data, 66.0, Random(1))
    print(evaluation.summary())

    # load a dataset incrementally and build classifier incrementally
    helper.print_title("Build classifier incrementally on iris")
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    iris_inc = loader.load_file(iris_file, incremental=True)
    iris_inc.class_is_last()
    classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")
    classifier.build_classifier(iris_inc)
    for inst in loader:
        classifier.update_classifier(inst)
    print(classifier)

    # construct meta-classifiers
    helper.print_title("Meta classifiers")
    # generic FilteredClassifier instantiation
    print("generic FilteredClassifier instantiation")
    meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.FilteredClassifier")
    meta.classifier = Classifier(classname="weka.classifiers.functions.LinearRegression")
    flter = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    flter.options = ["-R", "first"]
    meta.set_property("filter", flter.jobject)
    print(meta.to_commandline())
    # direct FilteredClassifier instantiation
    print("direct FilteredClassifier instantiation")
    meta = FilteredClassifier()
    meta.classifier = Classifier(classname="weka.classifiers.functions.LinearRegression")
    flter = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    flter.options = ["-R", "first"]
    meta.filter = flter
    print(meta.to_commandline())
    # generic Vote
    print("generic Vote instantiation")
    meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
    classifiers = [
        Classifier(classname="weka.classifiers.functions.SMO"),
        Classifier(classname="weka.classifiers.trees.J48")
    ]
    meta.classifiers = classifiers
    print(meta.to_commandline())

    # cross-validate nominal classifier
    helper.print_title("Cross-validating NaiveBayes on diabetes")
    diabetes_file = helper.get_data_dir() + os.sep + "diabetes.arff"
    helper.print_info("Loading dataset: " + diabetes_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    diabetes_data = loader.load_file(diabetes_file)
    diabetes_data.class_is_last()
    classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    pred_output = PredictionOutput(
        classname="weka.classifiers.evaluation.output.prediction.PlainText", options=["-distribution"])
    evaluation = Evaluation(diabetes_data)
    evaluation.crossvalidate_model(classifier, diabetes_data, 10, Random(42), output=pred_output)
    print(evaluation.summary())
    print(evaluation.class_details())
    print(evaluation.matrix())
    print("areaUnderPRC/0: " + str(evaluation.area_under_prc(0)))
    print("weightedAreaUnderPRC: " + str(evaluation.weighted_area_under_prc))
    print("areaUnderROC/1: " + str(evaluation.area_under_roc(1)))
    print("weightedAreaUnderROC: " + str(evaluation.weighted_area_under_roc))
    print("avgCost: " + str(evaluation.avg_cost))
    print("totalCost: " + str(evaluation.total_cost))
    print("confusionMatrix: " + str(evaluation.confusion_matrix))
    print("correct: " + str(evaluation.correct))
    print("pctCorrect: " + str(evaluation.percent_correct))
    print("incorrect: " + str(evaluation.incorrect))
    print("pctIncorrect: " + str(evaluation.percent_incorrect))
    print("unclassified: " + str(evaluation.unclassified))
    print("pctUnclassified: " + str(evaluation.percent_unclassified))
    print("coverageOfTestCasesByPredictedRegions: " + str(evaluation.coverage_of_test_cases_by_predicted_regions))
    print("sizeOfPredictedRegions: " + str(evaluation.size_of_predicted_regions))
    print("falseNegativeRate: " + str(evaluation.false_negative_rate(1)))
    print("weightedFalseNegativeRate: " + str(evaluation.weighted_false_negative_rate))
    print("numFalseNegatives: " + str(evaluation.num_false_negatives(1)))
    print("trueNegativeRate: " + str(evaluation.true_negative_rate(1)))
    print("weightedTrueNegativeRate: " + str(evaluation.weighted_true_negative_rate))
    print("numTrueNegatives: " + str(evaluation.num_true_negatives(1)))
    print("falsePositiveRate: " + str(evaluation.false_positive_rate(1)))
    print("weightedFalsePositiveRate: " + str(evaluation.weighted_false_positive_rate))
    print("numFalsePositives: " + str(evaluation.num_false_positives(1)))
    print("truePositiveRate: " + str(evaluation.true_positive_rate(1)))
    print("weightedTruePositiveRate: " + str(evaluation.weighted_true_positive_rate))
    print("numTruePositives: " + str(evaluation.num_true_positives(1)))
    print("fMeasure: " + str(evaluation.f_measure(1)))
    print("weightedFMeasure: " + str(evaluation.weighted_f_measure))
    print("unweightedMacroFmeasure: " + str(evaluation.unweighted_macro_f_measure))
    print("unweightedMicroFmeasure: " + str(evaluation.unweighted_micro_f_measure))
    print("precision: " + str(evaluation.precision(1)))
    print("weightedPrecision: " + str(evaluation.weighted_precision))
    print("recall: " + str(evaluation.recall(1)))
    print("weightedRecall: " + str(evaluation.weighted_recall))
    print("kappa: " + str(evaluation.kappa))
    print("KBInformation: " + str(evaluation.kb_information))
    print("KBMeanInformation: " + str(evaluation.kb_mean_information))
    print("KBRelativeInformation: " + str(evaluation.kb_relative_information))
    print("SFEntropyGain: " + str(evaluation.sf_entropy_gain))
    print("SFMeanEntropyGain: " + str(evaluation.sf_mean_entropy_gain))
    print("SFMeanPriorEntropy: " + str(evaluation.sf_mean_prior_entropy))
    print("SFMeanSchemeEntropy: " + str(evaluation.sf_mean_scheme_entropy))
    print("matthewsCorrelationCoefficient: " + str(evaluation.matthews_correlation_coefficient(1)))
    print("weightedMatthewsCorrelation: " + str(evaluation.weighted_matthews_correlation))
    print("class priors: " + str(evaluation.class_priors))
    print("numInstances: " + str(evaluation.num_instances))
    print("meanAbsoluteError: " + str(evaluation.mean_absolute_error))
    print("meanPriorAbsoluteError: " + str(evaluation.mean_prior_absolute_error))
    print("relativeAbsoluteError: " + str(evaluation.relative_absolute_error))
    print("rootMeanSquaredError: " + str(evaluation.root_mean_squared_error))
    print("rootMeanPriorSquaredError: " + str(evaluation.root_mean_prior_squared_error))
    print("rootRelativeSquaredError: " + str(evaluation.root_relative_squared_error))
    print("prediction output:\n" + str(pred_output))
    plot_cls.plot_roc(
        evaluation, title="ROC diabetes",
        class_index=range(0, diabetes_data.class_attribute.num_values), wait=False)
    plot_cls.plot_prc(
        evaluation, title="PRC diabetes",
        class_index=range(0, diabetes_data.class_attribute.num_values), wait=False)

    # train 2nd classifier on diabetes dataset
    classifier2 = Classifier(classname="weka.classifiers.trees.RandomForest")
    evaluation2 = Evaluation(diabetes_data)
    evaluation2.crossvalidate_model(classifier2, diabetes_data, 10, Random(42))
    plot_cls.plot_rocs({"NB": evaluation, "RF": evaluation2}, title="ROC diabetes", class_index=0, wait=False)
    plot_cls.plot_prcs({"NB": evaluation, "RF": evaluation2}, title="PRC diabetes", class_index=0, wait=False)

    # load a numeric dataset
    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    bolts_data = loader.load_file(bolts_file)
    bolts_data.class_is_last()

    # build a classifier and output model
    helper.print_title("Training LinearRegression on bolts")
    classifier = Classifier(classname="weka.classifiers.functions.LinearRegression", options=["-S", "1", "-C"])
    classifier.build_classifier(bolts_data)
    print(classifier)

    # cross-validate numeric classifier
    helper.print_title("Cross-validating LinearRegression on bolts")
    classifier = Classifier(classname="weka.classifiers.functions.LinearRegression", options=["-S", "1", "-C"])
    evaluation = Evaluation(bolts_data)
    evaluation.crossvalidate_model(classifier, bolts_data, 10, Random(42))
    print(evaluation.summary())
    print("correlationCoefficient: " + str(evaluation.correlation_coefficient))
    print("errorRate: " + str(evaluation.error_rate))
    helper.print_title("Header - bolts")
    print(str(evaluation.header))
    helper.print_title("Predictions on bolts")
    for index, pred in enumerate(evaluation.predictions):
        print(str(index+1) + ": " + str(pred) + " -> error=" + str(pred.error))
    plot_cls.plot_classifier_errors(evaluation.predictions, wait=False)

    # train 2nd classifier and show errors in same plot
    classifier2 = Classifier(classname="weka.classifiers.functions.SMOreg")
    evaluation2 = Evaluation(bolts_data)
    evaluation2.crossvalidate_model(classifier2, bolts_data, 10, Random(42))
    plot_cls.plot_classifier_errors({"LR": evaluation.predictions, "SMOreg": evaluation2.predictions}, wait=False)

    # learning curve
    cls = [
        Classifier(classname="weka.classifiers.trees.J48"),
        Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")]
    plot_cls.plot_learning_curve(
        cls, diabetes_data, increments=0.05, label_template="[#] !", metric="percent_correct", wait=True)

    # access classifier's Java API
    labor_file = helper.get_data_dir() + os.sep + "labor.arff"
    helper.print_info("Loading dataset: " + labor_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    labor_data = loader.load_file(labor_file)
    labor_data.class_is_last()

    helper.print_title("Using JRip's Java API to access rules")
    jrip = Classifier(classname="weka.classifiers.rules.JRip")
    jrip.build_classifier(labor_data)
    rset = jrip.jwrapper.getRuleset()
    for i in range(rset.size()):
        r = rset.get(i)
        print(str(r.toString(labor_data.class_attribute.jobject)))


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
