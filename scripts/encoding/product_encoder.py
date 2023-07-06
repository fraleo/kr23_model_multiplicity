from gurobipy import Model, GRB
import numpy as np

from scripts.encoding.constants import ReluNodeEncoding
from scripts.encoding.network_model import ReluLayer, LinearLayer
from scripts.encoding.specification import NAryConjFormula, Constraint


class ProductEncoder:

    EPSILON = 1e-5

    @staticmethod
    def get_real_variable_name(layer_n, node_n):
        return "R{}-{}".format(layer_n, node_n)

    @staticmethod
    def get_real_variable_name_m(model_n, layer_n, node_n):
        return "R{}-{}-{}".format(model_n, layer_n, node_n)

    @staticmethod
    def get_binary_variable_name(layer_n, node_n):
        return "B{}-{}".format(layer_n, node_n)

    @staticmethod
    def get_binary_variable_name_m(model_n, layer_n, node_n):
        return "B{}-{}".format(model_n, layer_n, node_n)

    def encode(self, neural_networks, var_networks, specification=None):
        gmodel = Model()

        variables = self._add_real_variables(gmodel, var_networks)

        self._add_network_constraints(gmodel, variables, neural_networks, var_networks)

        if specification is not None:
            self._add_output_specification_constraints(gmodel, variables, specification)
            self._add_objective_function(gmodel, variables, specification)

        return gmodel

    def _add_real_variables(self, gmodel, var_networks):
        # This structure stores all variables
        # The first element is the array of input variables
        # The second element is an array of all layer variables for the first model
        #   i.e., [[layer1 vars], [layer2 vars], ..]
        # The third element is an array of all layer variables for the second model
        # Etc
        variables = []

        # We assume that the input layer is the same for all models
        input_layer = var_networks[0].layers[0]
        input_variables = self.create_gurobi_variables(gmodel, input_layer, 0)
        variables.append(np.array(input_variables))

        # For each model, we create and add variables layer by layer
        for model_n, var_network in enumerate(var_networks):
            layer_n = 1

            model_variables = []
            for layer in var_network.layers[1:]:
                layer_vars = self.create_gurobi_variables_m(gmodel, layer, layer_n, model_n)

                layer_n += 1
                model_variables.append(np.array(layer_vars))

            variables.append(model_variables)

        gmodel.update()

        return variables

    def create_gurobi_variables(self, gmodel, layer, layer_n):
        lower_bounds = layer.get_bounds().get_lower()
        upper_bounds = layer.get_bounds().get_upper()

        gurobi_layer_vars = [
            gmodel.addVar(name=self.get_real_variable_name(layer_n, node_n),
                          lb=lower_bounds[node_n],
                          ub=upper_bounds[node_n])
            for node_n in range(layer.size)]

        return gurobi_layer_vars

    def create_gurobi_variables_m(self, gmodel, layer, layer_n, model_n):
        lower_bounds = layer.get_bounds().get_lower()
        upper_bounds = layer.get_bounds().get_upper()

        gurobi_layer_vars = [
            gmodel.addVar(name=self.get_real_variable_name_m(model_n, layer_n, node_n),
                          lb=lower_bounds[node_n],
                          ub=upper_bounds[node_n])
            for node_n in range(layer.size)]

        return gurobi_layer_vars

    def _add_network_constraints(self, gmodel, variables, neural_networks, var_networks):

        for var_network, model_n in zip(var_networks, range(len(var_networks))):

            model_variables = list([variables[0]]) + list(variables[model_n + 1])

            for layer_n in range(1, len(var_network.layers)):
                var_layer = var_network.layers[layer_n]

                network_layer = neural_networks[model_n].get_layer(layer_n)

                if isinstance(network_layer, ReluLayer):
                    self._add_relu_layer_constraints_m(gmodel, model_variables, network_layer, var_layer, model_n)
                elif isinstance(network_layer, LinearLayer):
                    self._add_linear_layer_constraints(gmodel, model_variables, network_layer, var_layer)
                else:
                    raise Exception("Not supported network layer type", network_layer)

    @staticmethod
    def _add_relu_layer_constraints_m(gmodel, gurobi_variables, network_layer, var_layer, model_n):
        """
        This methods adds MILP constraints of the ReLU nodes depending on their particular encoding.
        The constraints also include the linear transformation (computed as dot_product below).

        The encoding can be

          - Identity, in = out
          - Zero, out = 0
          - BigM, the standard BigM encoding

        and various relaxations
          - Linear Relaxation (as in Ehler 2017)
          - Quadratic Relaxation (not likely to be used, the quadratic version of the above)
          - Under Approximations, out is asserted as = 0 or >=0
        """
        dot_product = network_layer.get_dot_product(gurobi_variables[var_layer.index - 1])

        preact_lower_bounds = var_layer.get_preactivation_bounds().get_lower()
        preact_upper_bounds = var_layer.get_preactivation_bounds().get_upper()

        encodings = var_layer.get_encodings()

        for node_n in range(var_layer.size):
            node_var = gurobi_variables[var_layer.index][node_n]

            if encodings[node_n] == ReluNodeEncoding.IDENTITY:
                gmodel.addConstr(node_var == dot_product[node_n])

            elif encodings[node_n] == ReluNodeEncoding.ZERO:
                gmodel.addConstr(node_var == 0)

            elif encodings[node_n] == ReluNodeEncoding.BIGM:
                """
                Add the binary variable for the BIGM encoding
                """
                delta = gmodel.addVar(name=ProductEncoder.get_binary_variable_name_m(model_n, var_layer.index, node_n),
                                      vtype=GRB.BINARY)
                gmodel.update()

                """
                The BIG-M constraints
                """
                gmodel.addConstr(node_var >= dot_product[node_n])
                gmodel.addConstr(node_var <= dot_product[node_n] - preact_lower_bounds[node_n] * (1 - delta))
                gmodel.addConstr(node_var <= preact_upper_bounds[node_n] * delta)

            else:
                raise Exception("Not supported encoding type", encodings[node_n])

    @staticmethod
    def _add_linear_layer_constraints(gmodel, gurobi_variables, network_layer, var_layer):
        dot_product = network_layer.get_dot_product(gurobi_variables[var_layer.index - 1])

        for node_n in range(var_layer.size):
            gmodel.addConstr(gurobi_variables[var_layer.index][node_n] == dot_product[node_n])

    def _add_objective_function(self, gmodel, variables, specification):
        input_vars = variables[0]
        input_values = specification.input

        diff_vars = []
        for node_n, input_var in enumerate(input_vars):
                delta = gmodel.addVar(vtype=GRB.BINARY, name=self.get_binary_variable_name(0, node_n))
                diff = gmodel.addVar(name="diff{}".format(node_n), lb=0, ub=1)
                gmodel.addConstr((delta == 1) >> (diff == input_var - input_values[node_n]))
                gmodel.addConstr((delta == 0) >> (diff == input_values[node_n] - input_var))

                diff_vars.append(diff)

        import gurobipy as gp
        gmodel.setObjective(gp.quicksum(diff_vars), GRB.MINIMIZE)

    def _add_output_specification_constraints(self, gmodel, variables, specification):
        output_variables = [model_variables[-1] for model_variables in variables[1:]]
        last_layer_n = len(variables[1])

        argmax_variables, constrs = self._get_argmax_variables(gmodel, output_variables, last_layer_n)
        for c in constrs:
            gmodel.addConstr(c)

        # Not negating the output formula here like in the normal encoding
        # TODO: Need to change it in the default encoder as well
        # This logic should be in the specification itself
        output_constraints = specification.get_output_constraints()
        constrs = self._get_constrs_recursively(gmodel, output_constraints, argmax_variables)

        for c in constrs:
            gmodel.addConstr(c)
        gmodel.update()

    def _get_argmax_variables(self, gmodel, output_variables, last_layer_n):
        argmax_variables = []
        constrs = []
        for model_n, model_output_variables in enumerate(output_variables):
            if len(model_output_variables) == 2:
                var0 = model_output_variables[0]
                var1 = model_output_variables[1]
                upper = 1
                lower = 0
                if var0.lb > var1.ub:
                    upper = 0
                if var1.lb > var0.ub:
                    lower = 1

                delta = gmodel.addVar(vtype=GRB.BINARY, lb=lower, ub=upper,
                                      name=self.get_binary_variable_name_m(model_n, last_layer_n, 0))
                constrs.append((delta == 1) >> (var1 >= var0 + self.EPSILON))
                constrs.append((delta == 0) >> (var0 >= var1 + self.EPSILON))

                argmax_variables.append(delta)

            else:
                raise Exception("Non binary classification models are not currently supported")

        return argmax_variables, constrs


    def _get_constrs_recursively(self, gmodel, formula, vars):
        if isinstance(formula, Constraint):
            # Note: Constraint is a leaf (terminating) node.
            return [formula.get_atomic_constraint(vars)]

        if isinstance(formula, NAryConjFormula):
            constrs = []
            for subformula in formula.clauses:
                constrs += self._get_constrs_recursively(gmodel, subformula, vars)

            return constrs

        raise Exception("unexpected formula", formula)

