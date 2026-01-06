#---------------------------------------------------------------
# Name: fall_detection_algorithms.py
# Purpose: Including tensorflow/tensorflow lite/simple logic algorithm
# Author: Nhan Cao - nhan.cao@umconnect.umt.edu
# Last Updated: 03/21/2023
# Python Version: 3.9.16
# Bazel Version: 6.1.1
#---------------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
# from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tensorflow.keras.models import load_model

# Original Tensorflow version of the fall detection model
class StandardModel(object):
    def __init__(self, modelPath=None, features=8, windowSize=10, kernelSize=4, layersActivation='relu', 
                 finalActivation='softmax', dropOutRate=0.2, poolSize=2, classes=2, verbose=False):
        self.windowSize = windowSize
        self.verbose = verbose

        if modelPath is None:
            # Define a new model
            self.model = Sequential()
            
            self.model.add(Conv2D(8, (kernelSize, 1), input_shape=(self.windowSize, features, 1), activation=layersActivation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropOutRate))
            self.model.add(MaxPooling2D(pool_size=(poolSize, 1)))

            # Second Conv2D layer
            self.model.add(Conv2D(16, (kernelSize, 1), activation=layersActivation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropOutRate))
            self.model.add(MaxPooling2D(pool_size=(poolSize, 1)))

            # Third Conv2D layer
            self.model.add(Conv2D(32, (kernelSize, 1), activation=layersActivation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropOutRate))
            self.model.add(MaxPooling2D(pool_size=(poolSize, 1)))

            # Flatten and Dense layers
            self.model.add(Flatten())
            self.model.add(Dense(16, activation=layersActivation))
            self.model.add(Dropout(dropOutRate))
            self.model.add(Dense(classes, activation=finalActivation))
        else:
            self.model = load_model(modelPath)
            
        if self.verbose:
            self.model.summary()

    def train(self, trainData, validationData, batchSize=100, epochs=10):
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        X_train, y_train = trainData
        X_val, y_val = validationData
        self.model.fit(x=X_train, y=y_train, batch_size=batchSize, epochs=epochs, validation_data=(X_val, y_val))


    def predict_proba(self, data):
        return self.model.predict(data)
    
    def predict(self, data):
        start_time = time.time()
        ret = np.where(self.predict_proba(data) >= 0.5, 1, 0)
        end_time = time.time()
        inference_time_ms = (end_time - start_time)
        if not self.verbose:
            print("Inference time: {:.2f} ms".format(inference_time_ms))
        return ret
    
    def score(self, testData):
        X_test, y_test = testData
        loss_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        if self.verbose:
            print(loss_metrics)
            print(self.model.metrics_names)
        accuracy = loss_metrics[1]
        return accuracy
    
    def save(self, modelDir, modelName):
        self.model.save(f'{modelDir}/{modelName}.h5')

    def representative_dataset(self):
        # for data in self.converted_data[:1000]:  # Using the first 1000 samples for better representation
        #     print(data.shape)
        yield [self.converted_data.astype(np.float32)]

    def load_data(self, data):
        self.converted_data = data
    
    def tfline_convert(self, tflitedir, tflitename):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply model optinmization here
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize the model
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        
        model_tflite = converter.convert()
        printStatus = Path(f'{tflitedir}/{tflitename}.tflite').write_bytes(model_tflite)
        if self.verbose:
            print(printStatus)
        # Also convert to the tflite binary to c/c++ header file for future use
        # os.system(f'xxd -i {tflitedir}/{tflitename}.tflite > {tflitedir}/{tflitename}.h')
        # Since BAZEL will run in the cache directory, then running the above command will
        # cause a very long name of variable in the header file, we can temporarily modify
        # the header file after running the above command


# Tensorflow lite version of the fall detection model, re-utilized the skeleton from
# hello world example
# class TfliteModel(object):
#     def __init__(self, modelPath, tflm=False, verbose=False):
#         self.verbose = verbose
#         self.modelPath = modelPath
        
#         if tflm:
#             self.tflm_interpreter = tflm_runtime.Interpreter.from_file(self.modelPath)
#         else: # Use the standard tensorflow lite instead
#             # load TFLite interpreter and allocate tensor
#             self.tflite_interpreter = tf.lite.Interpreter(
#                 model_path=self.modelPath,
#                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.
#                 BUILTIN_REF,
#             )
#             self.tflite_interpreter.allocate_tensors()

#     # Re-utilize from hello_world example, tinyML project
#     def invoke_tflm_interpreter(self, input_shape, interpreter, x_value, input_index,
#                                 output_index):
#         input_data = np.reshape(x_value, input_shape)
#         interpreter.set_input(input_data, input_index)
#         interpreter.invoke()
#         y_quantized = np.reshape(interpreter.get_output(output_index), -1)[0]
#         return y_quantized

#     # Re-utilize from hello_world example, tinyML project but change
#     # the way of getting y_output since the standard structure of model
#     # will have 2 output of fall and not-fall, two-unit dense layer with 
#     # softmax activation, which means that the output of the model should 
#     # be a probability distribution over two classes
#     def invoke_tflite_interpreter(self, input_shape, interpreter, x_value, input_index,
#                                     output_index):
#         input_data = np.reshape(x_value, input_shape)
#         interpreter.set_tensor(input_index, input_data)
#         interpreter.invoke()
#         tflite_output = interpreter.get_tensor(output_index)
#         y_output = np.argmax(tflite_output, axis=-1)[0]  # This line has been corrected
#         return y_output

#     # Re-utilize from hello_world example, tinyML project
#     def get_tflm_prediction(self, x_values):
#         # Create the tflm interpreter
#         input_shape = np.array(self.tflm_interpreter.get_input_details(0).get('shape'))

#         y_predictions = np.empty(x_values.size, dtype=np.float32)

#         for i, x_value in enumerate(x_values):
#             y_predictions[i] = self.invoke_tflm_interpreter(input_shape,
#                                                         self.tflm_interpreter,
#                                                         x_value,
#                                                         input_index=0,
#                                                         output_index=0)
#         return y_predictions

#     # Re-utilize from hello_world example, tinyML project
#     def get_tflite_prediction(self, x_values):
#         input_details = self.tflite_interpreter.get_input_details()[0]
#         output_details = self.tflite_interpreter.get_output_details()[0]
        
#         input_shape = np.array(input_details.get('shape'))

#         y_predictions = np.empty(len(x_values), dtype=np.float32)
        
#         for i, x_value in enumerate(x_values):
#             start_time = time.time()
#             y_predictions[i] = self.invoke_tflite_interpreter(
#                 input_shape,
#                 self.tflite_interpreter,
#                 x_value,
#                 input_details['index'],
#                 output_details['index'],
#             )
#             end_time = time.time()
#             inference_time_ms = (end_time - start_time)
#             if self.verbose:
#                 print("Inference time: {:.2f} ms".format(inference_time_ms))
#         return y_predictions
    
#     # Evaluate the accurary compared to available labels
#     def evaluate_accuracy(self, x_values, y_true):
#         if hasattr(self, 'tflm_interpreter'):
#             y_pred = self.get_tflm_prediction(x_values)
#         else:
#             y_pred = self.get_tflite_prediction(x_values)
#         # Round the predicted values to the nearest integer (0 or 1)
#         y_pred_rounded = np.round(y_pred)
#         # Calculate the accuracy by comparing the true labels and predicted labels
#         accuracy = np.mean(y_pred_rounded == y_true)
#         return accuracy
    
#     # Get the prediction of fall or not-fall
#     def detectFall(self, data):        
#         return np.round(self.get_tflite_prediction(data))