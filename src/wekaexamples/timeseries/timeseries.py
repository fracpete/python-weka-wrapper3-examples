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

# timeseries.py
# Copyright (C) 2021-2022 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.timeseries import TSEvaluation, TSEvalModule, WekaForecaster
from weka.classifiers import Classifier
from weka.core.classes import serialization_write, serialization_read


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    airline_file = helper.get_data_dir() + os.sep + "airline.arff"
    helper.print_info("Loading dataset: " + airline_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    airline_data = loader.load_file(airline_file)
    airline_data.class_is_last()

    # available evaluation modules
    helper.print_title("Evaluation modules")
    modules = TSEvalModule.module_list()
    helper.print_info("Available modules")
    for module in modules:
        print("-" + str(module))
    helper.print_info("Loading module by name")
    print(TSEvalModule.module("MAE"))

    # evaluate forecaster
    helper.print_title("Evaluate forecaster")
    forecaster = WekaForecaster()
    forecaster.fields_to_forecast = ["passenger_numbers"]
    forecaster.base_forecaster = Classifier(classname="weka.classifiers.functions.LinearRegression")
    forecaster.tslag_maker.timestamp_field = "Date"
    forecaster.tslag_maker.adjust_for_variance = False
    forecaster.tslag_maker.include_powers_of_time = True
    forecaster.tslag_maker.include_timelag_products = True
    forecaster.tslag_maker.remove_leading_instances_with_unknown_lag_values = False
    forecaster.tslag_maker.add_month_of_year = True
    forecaster.tslag_maker.add_quarter_of_year = True
    print("algorithm name: " + str(forecaster.algorithm_name))
    print("command-line: " + forecaster.to_commandline())
    print("lag maker: " + forecaster.tslag_maker.to_commandline())

    evaluation = TSEvaluation(airline_data, 0.0)
    evaluation.evaluate_on_training_data = False
    evaluation.evaluate_on_test_data = False
    evaluation.prime_window_size = forecaster.tslag_maker.max_lag
    evaluation.prime_for_test_data_with_test_data = True
    evaluation.rebuild_model_after_each_test_forecast_step = False
    evaluation.forecast_future = True
    evaluation.horizon = 20
    evaluation.evaluation_modules = "MAE,RMSE"
    evaluation.evaluate(forecaster)
    print(evaluation)
    if evaluation.evaluate_on_training_data or evaluation.evaluate_on_test_data:
        print(evaluation.summary())
    if evaluation.evaluate_on_training_data:
        print("Predictions (training data): " + evaluation.predictions_for_training_data(1).summary)
    if evaluation.evaluate_on_test_data:
        print("Predictions (test data): " + evaluation.predictions_for_test_data(1).summary)
        preds = evaluation.predictions_for_test_data(1)
        print("Counts for targets: " + str(preds.counts_for_targets()))
        print("Errors for target 'passenger_numbers': " + str(preds.errors_for_target("passenger_numbers")))
        print("Errors for all targets: " + str(preds.predictions_for_all_targets()))
    if evaluation.training_data is not None:
        print("Future forecasts (training)\n" + evaluation.print_future_forecast_on_training_data(forecaster))
    if evaluation.test_data is not None:
        print("Future forecasts (test)\n" + evaluation.print_future_forecast_on_test_data(forecaster))
    if evaluation.evaluate_on_training_data:
        print(evaluation.print_predictions_for_training_data("Predictions (training)", "passenger_numbers", 1))
    if evaluation.evaluate_on_test_data:
        print(evaluation.print_predictions_for_test_data("Predictions (test)", "passenger_numbers", 1))

    # build forecaster
    helper.print_title("Build/use forecaster")
    airline_train, airline_test = airline_data.train_test_split(90.0)
    forecaster = WekaForecaster()
    forecaster.fields_to_forecast = ["passenger_numbers"]
    forecaster.base_forecaster = Classifier(classname="weka.classifiers.functions.LinearRegression")
    forecaster.fields_to_forecast = "passenger_numbers"
    forecaster.build_forecaster(airline_train)
    num_prime_instances = 12
    airline_prime = Instances.copy_instances(airline_train, airline_train.num_instances - num_prime_instances, num_prime_instances)
    forecaster.prime_forecaster(airline_prime)
    num_future_forecasts = airline_test.num_instances
    preds = forecaster.forecast(num_future_forecasts)
    print("Actual,Predicted,Error")
    for i in range(num_future_forecasts):
        actual = airline_test.get_instance(i).get_value(0)
        predicted = preds[i][0].predicted
        error = actual - predicted
        print("%f,%f,%f" % (actual, predicted, error))

    # serialization (if supported)
    helper.print_title("Serialization")
    model_file = helper.get_tmp_dir() + "/base.model"
    if forecaster.base_model_has_serializer:
        forecaster.save_base_model(model_file)
        forecaster2 = WekaForecaster()
        forecaster2.load_base_model(model_file)
        print(forecaster2.to_commandline())
    else:
        print("Base model has no serializer, falling back to generic serialization")
        serialization_write(model_file, forecaster.base_forecaster)
        cls = Classifier(jobject=serialization_read(model_file))
        print(cls.to_commandline())

    # state management
    helper.print_title("State")
    model_file = helper.get_tmp_dir() + "/state.ser"
    if forecaster.uses_state:
        forecaster.serialize_state(model_file)
        forecaster2 = WekaForecaster()
        forecaster2.load_serialized_state(model_file)
        print(forecaster2.to_commandline())
    else:
        print("Forecaster does not use state, falling back to generic serialization")
        serialization_write(model_file, forecaster)
        forecaster2 = WekaForecaster(jobject=serialization_read(model_file))
        print(forecaster2.to_commandline())


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
