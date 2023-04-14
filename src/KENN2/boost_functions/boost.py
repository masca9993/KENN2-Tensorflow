import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import softmax


def GodelBoostResiduum(clause_weight, antecedent_matrix, consequent_matrix):

    conjunction_val = tf.reduce_min(antecedent_matrix, axis=1)
    disjunction_val = tf.reduce_max(consequent_matrix, axis=1)
    formula_satisfied = tf.dtypes.cast(disjunction_val < conjunction_val, dtype=tf.float32)

    indices_consequent = tf.math.argmax(consequent_matrix, axis=1) + antecedent_matrix.shape[1]

    delta_mask = np.zeros((antecedent_matrix.shape[0], antecedent_matrix.shape[1] + consequent_matrix.shape[1]))
    delta_mask[np.arange(delta_mask.shape[0]), indices_consequent] = 1
    delta_mask = tf.convert_to_tensor(delta_mask, dtype=tf.float32)

    return tf.transpose(tf.math.minimum(clause_weight, conjunction_val-disjunction_val)
                        * tf.transpose(formula_satisfied)) * delta_mask

def GodelBoostResiduumApprox(clause_weight, antecedent_matrix, consequent_matrix):

    conjunction_val = -softplus(-1 * antecedent_matrix)  # we compute multivariable softminus, which is equal to -softplus(-x)
    disjunction_val = softplus(consequent_matrix)
    formula_satisfied = tf.dtypes.cast(disjunction_val < conjunction_val, dtype=tf.float32)
    delta_mask = tf.concat([tf.zeros_like(antecedent_matrix, dtype=tf.float32), softmax(consequent_matrix, axis=1)], 1)
    return tf.transpose(tf.math.minimum(clause_weight, conjunction_val-disjunction_val)
                        * tf.transpose(formula_satisfied)) * delta_mask


def softplus(matrix):
    return tf.reduce_logsumexp(matrix, 1)
