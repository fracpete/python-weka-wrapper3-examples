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
# Copyright (C) 2021 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.timeseries import TSForecaster, TSEvaluation, TSEvalModule, WekaForecaster
from weka.classifiers import Classifier


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    airline_file = helper.get_data_dir() + os.sep + "airline.arff"
    helper.print_info("Loading dataset: " + airline_file)
    loader = Loader("weka.core.converters.ArffLoader")
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
    forecaster = WekaForecaster(options=["-F", "passenger_numbers"])
    forecaster.base_forecaster = Classifier(classname="weka.classifiers.functions.LinearRegression")
    print("algorithm name: " + str(forecaster.algorithm_name))
    print("command-line: " + forecaster.to_commandline())

    evaluation = TSEvaluation(airline_data, 0.33)
    evaluation.evaluate_on_training_data = True
    evaluation.evaluate_on_test_data = True
    evaluation.prime_window_size = 10
    evaluation.prime_for_test_data_with_test_data = True
    evaluation.rebuild_model_after_each_test_forecast_step = False
    evaluation.forecast_future = True
    evaluation.horizon = 5
    evaluation.evaluation_modules = "MAE,RMSE"
    evaluation.evaluate(forecaster)
    print(evaluation)
    print(evaluation.summary())
    print("Predictions (training data): " + evaluation.predictions_for_training_data(evaluation.horizon - 1).summary)
    print("Predictions (test data): " + evaluation.predictions_for_test_data(evaluation.horizon - 1).summary)
    print("Future forecasts (training)\n" + evaluation.print_future_forecast_on_training_data(forecaster))
    print("Future forecasts (test)\n" + evaluation.print_future_forecast_on_test_data(forecaster))
    print(evaluation.print_predictions_for_training_data("Predictions (training)", "passenger_numbers", evaluation.horizon - 1))
    print(evaluation.print_predictions_for_test_data("Predictions (test)", "passenger_numbers", evaluation.horizon - 1))
    # for module in evaluation.evaluation_modules:
    #     print(module.eval_name + ": " + str(module.target_fields))

    # build forecaster
    helper.print_title("Build/use forecaster")
    airline_train, airline_test = airline_data.train_test_split(66)
    forecaster = WekaForecaster(options=["-F", "passenger_numbers"])
    forecaster.base_forecaster = Classifier(classname="weka.classifiers.functions.LinearRegression")
    forecaster.fields_to_forecast = "passenger_numbers"
    forecaster.build_forecaster(airline_train)
    forecaster.prime_forecaster(airline_test)
    steps = forecaster.forecast(10)
    for i, step in enumerate(steps):
        print("Step #" + str(i+1) + ": " + str(step))

    # serialization (if supported)
    helper.print_title("Serialization")
    if forecaster.base_model_has_serializer:
        model_file = helper.get_tmp_dir() + "/base.model"
        forecaster.save_base_model(model_file)
        forecaster2 = WekaForecaster(options=["-F", "passenger_numbers"])
        forecaster2.load_base_model(model_file)
        print(forecaster2)
    else:
        print("Base model has no serializer, skipping serialization I/O")

    # state management
    helper.print_title("State")
    if forecaster.uses_state:
        state_file = helper.get_tmp_dir() + "/state.ser"
        forecaster.serialize_state(state_file)
        forecaster2 = WekaForecaster(options=["-F", "passenger_numbers"])
        forecaster2.load_serialized_state(state_file)
        print(forecaster2)
    else:
        print("Forecaster does not use state, skipping state I/O")


if __name__ == "__main__":
    try:
        jvm.start(system_info=True, packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
