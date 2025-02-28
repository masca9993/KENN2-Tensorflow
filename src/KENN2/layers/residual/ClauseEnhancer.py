import tensorflow as tf
import numpy as np

from KENN2.layers.RangeConstraint import RangeConstraint



class ClauseEnhancer(tf.keras.layers.Layer):
    """Clause Enhancer layer
    """

    def __init__(self, available_predicates, clause_string, initial_clause_weight, save_training_data=False):
        """Initialize the clause.
        :param available_predicates: the list of all possible literals in a clause
        :param clause_string: a string representing a conjunction of literals. The format should be:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this sign is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial sign to the clause weight. Used if the clause weight is learned.
        """

        super(ClauseEnhancer, self).__init__()
        string = clause_string.split(':')

        self.original_string = string[1]
        self.string = string[1].replace(
            ',', 'v').replace('(', '').replace(')', '')

        if string[0] == '_':
            self.initial_weight = initial_clause_weight
            self.hard_clause = False
        else:
            self.initial_weight = int(string[0])
            self.hard_clause = True

        self.hard_clause = string[0] != '_'

        literals_list = string[1].split(',')
        self.number_of_literals = len(literals_list)

        self.gather_literal_indices = []
        self.scatter_literal_indices = []
        signs = []

        for literal in literals_list:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            literal_index = available_predicates.index(literal)
            self.gather_literal_indices.append(literal_index)
            self.scatter_literal_indices.append([literal_index])
            signs.append(sign)

        self.signs = np.array(signs, dtype=np.float32)
        self.clause_weight = None
        self.save_training_data = save_training_data

    def build(self, input_shape):
        """Build the layer
        :param input_shape: the input shape
        """
        self.clause_weight = self.add_weight(
            name='kernel',
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(
                value=self.initial_weight),
            constraint=RangeConstraint(),
            trainable=not self.hard_clause)

        super(ClauseEnhancer, self).build(input_shape)

    def grounded_clause(self, inputs):
        """Find the grounding of the clause
        :param inputs: the tensor containing predicates' pre activations for many objects
        :return: the grounded clause (a tensor with literals truth values)
        """

        selected_predicates = tf.gather(
            inputs, self.gather_literal_indices, axis=1)
        clause_matrix = selected_predicates * self.signs

        return clause_matrix

    def call(self, inputs, **kwargs):
        """Improve the satisfaction level of the clause.
        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: delta vector to be summed to the original pre-activation tensor to obtain an higher satisfaction of \
        the clause"""

        clause_matrix = self.grounded_clause(inputs)

        delta = self.signs * tf.nn.softmax(clause_matrix) * self.clause_weight

        return delta, self.scatter_literal_indices


class ClauseEnhancerImpl(tf.keras.layers.Layer):

    def __init__(self, available_predicates, formula_string, initial_clause_weight, boost_function, save_training_data=False):

        super(ClauseEnhancerImpl, self).__init__()
        weight_clause_split = formula_string.split(':')

        self.formula_string = weight_clause_split[1]
        weight_string = weight_clause_split[0]

        if weight_string == '_':
            self.initial_weight = initial_clause_weight
            self.hard_clause = False
        else:
            self.initial_weight = float(weight_string)
            self.hard_clause = True

        self.hard_clause = weight_string != '_'

        self.conorm_boost = boost_function

        # Split the formula in antecedent (before '->') and consequent (after '->')
        formula = self.formula_string.split('->')
        antecedent_conjunction = formula[0]
        consequent_disjunction = formula[1]

        # Semicolon represent conjunction of literals in the antecedent proposition
        antecedent_literals = antecedent_conjunction.split(';')
        consequent_literals = consequent_disjunction.split(',')

        self.scatter_literal_indices = []

        self.antecedent_literal_indices = []
        antecedent_signs = []

        # Do the same operation done in ClauseEnhancer separately for antecedent and consequent literals
        for literal in antecedent_literals:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            literal_index = available_predicates.index(literal)
            self.antecedent_literal_indices.append(literal_index)

            antecedent_signs.append(sign)

        self.consequent_literal_indices = []
        consequent_signs = []

        for literal in consequent_literals:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            literal_index = available_predicates.index(literal)
            self.consequent_literal_indices.append(literal_index)

            self.scatter_literal_indices.append([literal_index])

            consequent_signs.append(sign)

        self.signs = [np.array(antecedent_signs, dtype=np.float32), np.array(consequent_signs, dtype=np.float32)]
        self.clause_weight = None
        self.save_training_data = save_training_data

    def build(self, input_shape):
        """Build the layer
        :param input_shape: the input shape
        """
        self.clause_weight = self.add_weight(
            name='kernel',
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(
                value=self.initial_weight),
            constraint=RangeConstraint(),
            trainable=not self.hard_clause)

        super(ClauseEnhancerImpl, self).build(input_shape)

    def grounded_clause(self, inputs):
        """Find the grounding of the clause
        :param inputs: the tensor containing predicates' pre activations for many objects
        :return: the grounded clause (a tensor with literals truth values)
        """
        selected_antecedent_predicates = tf.gather(
            inputs, self.antecedent_literal_indices, axis=1)
        selected_consequent_predicates = tf.gather(
            inputs, self.consequent_literal_indices, axis=1)

        antecedent_matrix = selected_antecedent_predicates * self.signs[0]
        consequent_matrix = selected_consequent_predicates * self.signs[1]

        return antecedent_matrix, consequent_matrix

    def call(self, inputs, **kwargs):
        """Improve the satisfaction level of the clause.
        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: delta vector to be summed to the original pre-activation tensor to obtain an higher satisfaction of \
        the clause"""
        antecedent_matrix, consequent_matrix = self.grounded_clause(inputs)
        delta = self.conorm_boost(self.clause_weight, antecedent_matrix, consequent_matrix) * self.signs[1]

        return delta, self.scatter_literal_indices

