import tensorflow as tf
import numpy as np
from KENN2.layers.residual.KnowledgeEnhancer import KnowledgeEnhancer
from KENN2.layers.relational.Join import Join
from KENN2.layers.relational.GroupBy import GroupBy
from KENN2.boost_functions.boost import GodelBoostResiduum, GodelBoostResiduumApprox

class RelationalKENN(tf.keras.layers.Layer):

    def __init__(self, unary_predicates, binary_predicates, unary_clauses, binary_clauses, implication_clauses, activation=lambda x: x, initial_clause_weight=0.5, boost_function=GodelBoostResiduum, **kwargs):
        """Initialize the knowledge base.

        :param unary_predicates: the list of unary predicates names
        :param binary_predicates: the list of binary predicates names
        :param unary_clauses: a list of unary clauses. Each clause is a string on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        :param binary_clauses: a list of binary clauses
        :param activation: activation function
        :param initial_clause_weight: initial value for the cluase weight (if clause is not hard)
        """

        super(RelationalKENN, self).__init__(**kwargs)

        self.unary_predicates = unary_predicates
        self.n_unary = len(unary_predicates)
        self.unary_clauses = unary_clauses
        self.binary_predicates = binary_predicates
        self.binary_clauses = binary_clauses
        self.implication_unary_clauses = implication_clauses[0]   #implication unary clauses are passed as first element in the list of length 2
        self.implication_binary_clauses = implication_clauses[1]  #implication binary clauses are passed as first element in the list of length 2
        self.activation = activation
        self.initial_clause_weight = initial_clause_weight
        self.boost_function = boost_function

        self.unary_ke = None
        self.binary_ke = None
        self.implication_ke = None
        self.join = None
        self.group_by = None

    def build(self, input_shape):
        if len(self.unary_clauses) != 0:
            self.unary_ke = KnowledgeEnhancer(
                self.unary_predicates, self.unary_clauses, initial_clause_weight=self.initial_clause_weight)

        if len(self.binary_clauses) != 0:
            self.binary_ke = KnowledgeEnhancer(
                self.binary_predicates, self.binary_clauses, initial_clause_weight=self.initial_clause_weight)

        if len(self.implication_unary_clauses) != 0:
            self.implication_ke = KnowledgeEnhancer(
                self.unary_predicates, self.implication_unary_clauses, initial_clause_weight=self.initial_clause_weight,
                implication=True, boost_function=self.boost_function)

        if len(self.implication_binary_clauses) != 0:
            self.implication_ke = KnowledgeEnhancer(
                self.binary_predicates, self.implication_binary_clauses,
                initial_clause_weight=self.initial_clause_weight, implication=True, boost_function=self.boost_function)

        self.join = Join()
        self.group_by = GroupBy(self.n_unary)
        super(RelationalKENN, self).build(input_shape)

    def call(self, unary, binary, index1, index2, **kwargs):
        """Forward step of Kenn model for relational data.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index2: a vector containing the indices of the second object
        of the pair referred by binary tensor
        """

        if len(self.unary_clauses) != 0:
            deltas_sum = self.unary_ke(unary)
            u = unary + deltas_sum
        else:
            u = unary

        if len(self.binary_clauses) != 0 and len(binary) != 0:
            joined_matrix = self.join(u, binary, index1, index2)
            deltas_sum = self.binary_ke(joined_matrix)
            delta_up, delta_bp = self.group_by(
                u, binary, deltas_sum, index1, index2)
        else:
            delta_up = tf.zeros_like(u)
            delta_bp = tf.zeros_like(binary)

        if len(self.implication_unary_clauses) != 0:
            deltas_sum = self.implication_ke(unary)
            u_impl = u + deltas_sum
        else:
            u_impl = u

        if len(self.implication_binary_clauses) != 0:
            joined_matrix = self.join(u, binary, index1, index2)
            deltas_sum = self.implication_ke(joined_matrix)
            delta_impl_up, delta_impl_bp = self.group_by(
                u, binary, deltas_sum, index1, index2)
        else:
            delta_impl_up = tf.zeros_like(u)
            delta_impl_bp = tf.zeros_like(binary)

        return self.activation(u_impl + delta_up + delta_impl_up), self.activation(binary + delta_bp + delta_impl_bp)

    def get_config(self):
        config = super(RelationalKENN, self).get_config()
        config.update({'unary_predicates': self.unary_predicates})
        config.update({'unary_clauses': self.unary_clauses})
        config.update({'binary_predicates': self.binary_predicates})
        config.update({'binary_clauses': self.binary_clauses})
        config.update({'activation': self.activation})
        config.update({'initial_clause_weight': self.initial_clause_weight})

        return config
