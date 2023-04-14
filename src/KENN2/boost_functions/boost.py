import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import softmax


def GodelBoostResiduum(clause_weight, antecedent_matrix, consequent_matrix):

    conjunction_val = tf.reduce_min(antecedent_matrix, axis=1)
    disjunction_val = tf.reduce_max(consequent_matrix, axis=1)
    formula_unsatisfied = tf.dtypes.cast(disjunction_val < conjunction_val, dtype=tf.float32)

    indices_consequent = tf.math.argmax(consequent_matrix, axis=1)

    delta_mask = np.zeros((consequent_matrix.shape[0], consequent_matrix.shape[1]))
    delta_mask[np.arange(delta_mask.shape[0]), indices_consequent] = 1
    delta_mask = tf.convert_to_tensor(delta_mask, dtype=tf.float32)

    return tf.transpose(tf.math.minimum(clause_weight, conjunction_val-disjunction_val)
                        * tf.transpose(formula_unsatisfied)) * delta_mask

def GodelBoostResiduumApprox(clause_weight, antecedent_matrix, consequent_matrix):

    conjunction_val = -softplus(-1 * antecedent_matrix)  # we compute multivariable softminus, which is equal to -softplus(-x)
    disjunction_val = softplus(consequent_matrix)
    formula_unsatisfied = tf.dtypes.cast(disjunction_val < conjunction_val, dtype=tf.float32)

    return tf.transpose(tf.math.minimum(clause_weight, conjunction_val-disjunction_val)
                        * tf.transpose(formula_unsatisfied)) * softmax(consequent_matrix, axis=1)


def softplus(matrix):
    return tf.reduce_logsumexp(matrix, 1)
