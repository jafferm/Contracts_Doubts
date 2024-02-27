from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from contracts import contract, new_contract
from keras import backend as K
import numpy as np
from contract_checker_library import ContractException
#excel sheet row 10
@new_contract
def batch_norm_order(model):
    for i in range(1, len(model.layers) - 1):  # Iterate from the second layer to the second-to-last layer
     current_layer = model.layers[i]
     previous_layer = model.layers[i - 1]
     next_layer = model.layers[i + 1]

     if isinstance(current_layer, BatchNormalization):
            if isinstance(previous_layer, Dense) and not isinstance(next_layer, Dense):
                break
            else:
                    raise ContractException("Invalid layer configuration: The layer before Batch Normalization should be Dense(linear-layer), and the layer after should be non-linear.")
            
#excel sheet row 22
@new_contract
def check_reset_weights(model):
    initial_weights = tf.keras.models.load_model('initial_weights.h5').get_weights()
    #This loop iterates over the layers of the model and the corresponding initial weights loaded from the saved file.
    for layer, initial_weight in zip(model.layers, initial_weights):
            current_weight = layer.get_weights()
            #This line checks whether all elements of the current_weight tensor are equal to the initial_weight tensor using TensorFlow operations.
            if not tf.reduce_all(tf.equal(current_weight, initial_weight)):
                raise ContractException("Weights are not reset to initial weights.")


#excel sheet row 25
@new_contract
def check_BN_updateOps(model, X_train, y_train):
    # It starts a TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Initializes all the variables in the model 
        sess.run(tf.compat.v1.global_variables_initializer())

        """
        Retrieves the update operations from the model. These operations are responsible
        for updating the Batch Normalization statistics during training
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Runs the update operations and a forward pass through the model with the provided input data
        sess.run([update_ops, model.output], feed_dict={model.input: X_train, model.output: y_train})

        """
        Checks if any update operations were executed. If update_ops is empty,
        it raises a ContractException indicating that Batch Normalization statistics
        are not being updated during training.
         """
        if not update_ops:
            raise ContractException("Batch Normalization statistics are not updated during training. "
                                    "You need to manually add the update operations.")    
