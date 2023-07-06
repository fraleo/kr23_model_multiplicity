import copy
import sys

from keras.utils import to_categorical

from scripts.encoding.network_model import NetworkModel
from scripts.encoding.node_variable_network import NodeVariableNetwork
from scripts.encoding.product_encoder import ProductEncoder
from scripts.encoding.specification import RobustExplanationSpecification
from scripts.encoding.symb_backward_bounds_calc import SymbolicBackwardBoundsCalculator
from util import Stats, Dataset

sys.path.append('')

import numpy as np
from gurobipy import GRB
from timeit import default_timer as timer

from tensorflow import keras


import pandas as pd
from sklearn.model_selection import train_test_split


class ModelMultiplicityRobustness:
    INFEASIBLE = "Infeasible"
    OPTIMAL = "Optimal"
    TIMEOUT = "Timeout"
    INTERRUPTED = "Interrupted"

    STRING_RESULT = {GRB.INFEASIBLE: INFEASIBLE,
                     GRB.OPTIMAL: OPTIMAL,
                     GRB.TIME_LIMIT: TIMEOUT,
                     GRB.INTERRUPTED: INTERRUPTED}

    def setUp(self):

        self.bounds_calculator = SymbolicBackwardBoundsCalculator()
        self.encoder = ProductEncoder()

        self.number_of_inputs = 50

        self.multiplicity = 5
        self.multiplicity_step = 1
        self.multiplicity_start = 2

    def setUp_scalability(self):
        self.bounds_calculator = SymbolicBackwardBoundsCalculator()
        self.encoder = ProductEncoder()

        self.number_of_inputs = 30

        self.multiplicity = 50
        self.multiplicity_step = 5
        self.multiplicity_start = 10

    def test_german(self):

        print("=============== Testing German dataset ===============")
        paths = [f"../models/german/nn_german_seed_{n}.h5" for n in range(self.multiplicity)]

        df_train = pd.read_csv('../datasets/german/train.csv').iloc[:, 1:]
        df_test = pd.read_csv('../datasets/german/test.csv').iloc[:, 1:]

        # Extract feature names and target
        names = list(df_train.columns)
        feature_names = names[:-1]
        target = names[-1]

        # Convert to numpy for later use
        x_train, y_train = df_train[feature_names].to_numpy(), df_train[target].to_numpy()
        # y_train = to_categorical(y_train)

        x_test, y_test = df_test[feature_names].to_numpy(), df_test[target].to_numpy()
        y_test = to_categorical(y_test)

        self.run_multiple_robustness(paths, x_test, y_test, x_train)

    def test_diabetes(self):

        print("=============== Testing Diabetes dataset ===============")

        paths = [f"../models/diabetes/nn_diabetes_seed_{n}.h5" for n in range(self.multiplicity)]

        df = pd.read_csv('../datasets/diabetes/diabetes.csv')
        df = df.dropna()

        self.load_test_data_and_run_multiple_robustness(df, paths)

    def test_no2(self):

        print("=============== Testing no2 dataset ===============")

        paths = [f"../models/no2/nn_no2_seed_{n}.h5" for n in range(self.multiplicity)]

        df = pd.read_csv('../datasets/no2/no2.csv')
        df = df.dropna()
        df = df.replace(to_replace={'N': 0, 'P': 1})

        self.load_test_data_and_run_multiple_robustness(df, paths)

    def load_test_data_and_run_multiple_robustness(self, df, paths):
        continuous_features = list(df.columns)[:-1]

        # min max scale
        min_vals = np.min(df[continuous_features], axis=0)
        max_vals = np.max(df[continuous_features], axis=0)
        df_mm = Dataset.min_max_scale(df, continuous_features, min_vals, max_vals)

        # get X, y
        X, y = df_mm.drop(columns=['Outcome']), pd.DataFrame(df_mm['Outcome'])

        SPLIT = .2
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=SPLIT, shuffle=True,
                                                            random_state=0)

        x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
        # y_train = to_categorical(y_train)
        x_test, y_test = x_test.to_numpy(), y_test.to_numpy()
        y_test = to_categorical(y_test)
        self.run_multiple_robustness(paths, x_test, y_test, x_train)

    def run_multiple_robustness(self, paths, x_test, y_test, x_train):
        max_multiplicity = len(paths)

        all_keras_models = {n: keras.models.load_model(path) for n, path in enumerate(paths)}
        all_network_models = [NetworkModel() for _ in range(max_multiplicity)]
        for network_model, path in zip(all_network_models, paths):
            network_model.parse(path)

        stats = Stats(x_train, all_keras_models)

        total_time = 0

        for multiplicity in range(self.multiplicity_start, max_multiplicity + 1, self.multiplicity_step):
            count = 0
            input_index = 0

            keras_models = {n: model for n, model in all_keras_models.items() if n < multiplicity}
            network_models = all_network_models[:multiplicity]

            while count < self.number_of_inputs and input_index < len(x_test):
                input = x_test[input_index]
                label = np.argmax(y_test[input_index])

                # consider only correctly classified inputs
                outputs = [keras_model.predict(input.reshape(1, -1), batch_size=1, verbose=0).reshape(-1)
                           for keras_model in keras_models.values()]
                correctly_classified = [np.argmax(out) == label for out in outputs]

                if not (False in correctly_classified):
                    count += 1

                    total_time += self.find_cfx(input, input_index, label, keras_models, network_models, multiplicity, stats)

                input_index += 1

        print("Total time ", total_time, "\n")

    def find_cfx(self, input, input_index, label, keras_models, network_models, multiplicity, stats):
        # print("Index {}, label {}".format(input_index, label))

        specification = RobustExplanationSpecification(input.reshape(-1), label, multiplicity)

        start = timer()
        var_networks = []
        for network_model in network_models:
            var_network = NodeVariableNetwork()
            var_network.initialise_layers(network_model)
            var_networks.append(var_network)

            self.bounds_calculator.compute_bounds(network_model, var_network, specification.get_input_bounds())

        gmodel = self.encoder.encode(network_models, var_networks, specification)
        gmodel.setParam("OutputFlag", 0)
        gmodel.setParam("TimeLimit", 1800)
        gmodel.optimize()
        end = timer()

        if gmodel.status == GRB.OPTIMAL:
            # print("Found explanation robust to model multiplicity, distance ", gmodel.objVal)
            input_vars = [gmodel.getVarByName(self.encoder.get_real_variable_name(0, node_n))
                          for node_n in range(var_networks[0].layers[0].size)]
            input_vars_values = np.array(
                [input_vars[node_n].x for node_n in range(var_networks[0].layers[0].size)]
            ).reshape(1, -1)
            valid, dist, lof = stats._evaluate_explanation(input.reshape(1, -1), input_vars_values)#, keras_models)

        else:
            # print("No counterfactual explanation found")
            valid, dist, lof = False, -1, 0

        print("{}, {}, {}, {}, {:8.6f}, {}, {:9.4f}, {}".format(multiplicity, input_index, label, valid, dist, lof,
                                                                end - start, self.STRING_RESULT[gmodel.status]))
        sys.stdout.flush()
        return end - start


if __name__ == '__main__':
    test = ModelMultiplicityRobustness()
    test.setUp()

    """Applicability experiment for the three datasets"""
    test.test_german()
    test.test_diabetes()
    test.test_no2()

    """Scalability experiment for the credit dataset"""
    # test.setUp_scalability()
    # test.test_german()
