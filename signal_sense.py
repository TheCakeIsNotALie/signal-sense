import os
# Avoid Tensorflow yelling
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pe_extractor.cnn import generator_nsb
from copy import deepcopy
import numpy as np
from math import ceil
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import itertools
from keras import backend as K
import keras_tuner
from matplotlib.pyplot import cm
import io

threshold = 0.5

def compute_sample_weights(samples):
    indices, counts = np.unique(samples, return_counts=True)
    weight_fn = lambda x: 1/(counts[np.where(indices == x)[0][0]]) * (len(samples)/len(indices)/2)
    return np.array(list(map(weight_fn, samples)))

def accuracy_m(y_true, y_pred):
    # Determine the actual positives (values in y_true that are not 0)
    actual_positives = tf.not_equal(y_true, 0)
    
    # Determine the true positives (actual positives where y_pred > 0.5)
    true_positives = tf.logical_and(actual_positives, y_pred >= threshold)
    true_negatives = tf.logical_and(tf.logical_not(actual_positives), y_pred < threshold)

    correct_prediction_count = tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(true_negatives, tf.float32))

    return correct_prediction_count / (tf.cast(tf.size(y_true), tf.float32) + tf.keras.backend.epsilon())

def recall_m(y_true, y_pred):    
    # Determine the actual positives (values in y_true that are not 0)
    actual_positives = tf.not_equal(y_true, 0)
    
    # Determine the true positives (actual positives where y_pred > 0.5)
    true_positives = tf.logical_and(actual_positives, y_pred >= threshold)
    predicted_positives = y_pred >= threshold
    
    # Count the true positives and actual positives
    true_positive_count = tf.reduce_sum(tf.cast(true_positives, tf.float32))
    predicted_positives_count = tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
    
    # Calculate recall
    recall = true_positive_count / (predicted_positives_count + tf.keras.backend.epsilon())
    
    return recall

def precision_m(y_true, y_pred):
    # Determine the possible positives (values in y_true that are not 0)
    positives = tf.not_equal(y_true, 0)
    
    # Determine the true positives (possible positives where y_pred > 0.5)
    true_positives = tf.logical_and(positives, y_pred >= threshold)
    
    # Count the true positives and possible positives
    true_positive_count = tf.reduce_sum(tf.cast(true_positives, tf.float32))
    positive_count = tf.reduce_sum(tf.cast(positives, tf.float32))
    
    # Calculate precision
    precision = true_positive_count / (positive_count + tf.keras.backend.epsilon())
    
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def test_custom_metrics():
    # Custom metrics test
    # 5 positives, 6 negatives
    expected_results = tf.constant([[0,1,2,0,4,2,1,0,0,0,0]])
    # 3 true positives, 1 false positive, 2 false negatives, 5 true negatives
    # 11 total
    predictions = tf.constant([[0.1, 0.6, 0.2, 0.01, 0.7, 1, 0.4, 0.0, 0.1, 0.3, 2]])
    
    print(f"Expected results : ")
    print(f"\tAccuracy : 0.72, {accuracy_m(expected_results, predictions)}")
    print(f"\tPrecision : 0.6, {precision_m(expected_results, predictions)}")
    print(f"\tRecall : 3/4 - {recall_m(expected_results, predictions)}")
    print(f"\tF1 : 0.666 - {f1_m(expected_results, predictions)}")

def build_cnn_model(hp: keras_tuner.HyperParameters, window_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(window_size, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(5, 
                                   hp.Int("kernel_size", min_value=5, max_value=window_size, step=2), 
                                   activation=hp.Choice("activation",["sigmoid", "relu", "tanh"])
                                   ),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(window_size, activation='relu'),
        ]
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=[precision_m, recall_m, f1_m])
    
    return model


"""
Overlap should normally overlap window_size-1
"""
def reaggregate_windows_avg(windows, window_size, overlap):
    # Determine the step size between windows
    step_size = window_size - overlap
    
    # Calculate the length of the final signal
    total_length = (len(windows) - 1) * step_size + window_size
    
    # Initialize arrays to accumulate values and counts
    accumulated_values = np.zeros(total_length)
    count_contributions = np.zeros(total_length)
    
    # Accumulate values and counts
    for i, window in enumerate(windows):
        start = i * step_size
        end = start + window_size
        accumulated_values[start:end] += window
        count_contributions[start:end] += 1
    
    # Average the accumulated values by counts
    reaggregated_signal = accumulated_values / count_contributions
    
    return reaggregated_signal

def get_colormap(values):
    color = iter(cm.rainbow(np.linspace(0, 1, len(values))))
    colormap = {}
    for i in range(len(values)):
        colormap[str(values[i])] = next(color)
        
    return colormap

# Function to capture model summary as a string
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def config_permutations(configs):
    permutations = []
    
    # go through each key/value pair of the dictionary
    for key, values in configs.items():
        next_permutations = []
        
        # if the permutations array does not have anything in it yet just add the current config param to it
        if not permutations:
            for value in values:
                perm = {}
                perm[key] = value
                next_permutations.append(perm)
        
        # add the current key/values as a subsequent permutation to the existing configurations
        for r in itertools.product(permutations, values):
            perm = deepcopy(r[0])
            perm[key] = r[1]
            next_permutations.append(perm)
        
        # apply the next permutation to the current one
        permutations = next_permutations
            
    return permutations

def calculate_metrics(model, y, y_pred, sample_weight=None):
    # print(f"Calculating metrics {y.shape} {y_pred.shape}")
    
    y = tf.convert_to_tensor(y)
    y_pred = tf.convert_to_tensor(y_pred)
    if not sample_weight is None:
        sample_weight = tf.convert_to_tensor(sample_weight)
        
    # model.compute_loss(x=None, y=y, y_pred=y_pred, sample_weight=sample_weight)
    metrics = model.compute_metrics(x=None, y=y, y_pred=y_pred,
                                    sample_weight=sample_weight)

    # convert tensors variable to float
    for k,v in metrics.items():
        metrics[k] = float(v)
        
    return metrics

def run_tests(model : tf.keras.Model, configs: dict,  window_size: int):
    metrics_result = []
    
    # static generator settings
    n_sample = 100000
    n_sample_init = 0
    batch_size = 1
    shift_proba_bin = 30
    sigma_smooth_pe_ns = 0 #2.
    bin_size_ns = 0.5   
    sampling_rate_mhz = 200 #200 MHz is the sampling of terzina
    amplitude_gain=5.
    relative_gain_std=0.05
        
    sampling_period_s = 1 / (sampling_rate_mhz * 1e6)
    bin_per_sample = ceil(((sampling_period_s) * 1e9) / bin_size_ns)
    
    permutations = config_permutations(configs)
    print(f"Running {len(permutations)} tests on model")
    for i, config in enumerate(permutations):
        # dynamic configs
        pe_rate_mhz = config["pe_rate_mhz"]
        noise_lsb = config["noise_lsb"]
        
        # initialize the generator from the static/dynamic config variables
        gen = generator_nsb(
            n_event=None, batch_size=batch_size, n_sample=n_sample + n_sample_init,
            n_sample_init=n_sample_init, pe_rate_mhz=pe_rate_mhz,
            bin_size_ns=bin_size_ns, sampling_rate_mhz=sampling_rate_mhz,
            amplitude_gain=amplitude_gain, noise_lsb=noise_lsb,
            sigma_smooth_pe_ns=sigma_smooth_pe_ns, baseline=0,
            relative_gain_std=relative_gain_std, shift_proba_bin=shift_proba_bin, dtype=np.float64
        )

        # fetch the data from the generator
        data = next(gen)
        # since many bins are created per sample, sum them up to the same time scale as sample rate
        summed_bins = np.sum(data[1][0].reshape(-1, bin_per_sample), axis=1)
        
        print(f"RUN_{i}: config={config} | bps={bin_per_sample} | input={data[0][0].shape} | expected_output={summed_bins.shape}")

        sliced_data = sliding_window_view(data[0][0], window_size)
        sliced_results = sliding_window_view(summed_bins, window_size)

        prediction = model.predict(sliced_data, verbose=0)
        metrics = calculate_metrics(model, prediction, sliced_results)

        merged = {}
        merged.update(config)
        merged.update(metrics)
        metrics_result.append(merged)
        
        # print(f"RES_{i}: {metrics}")
    
    return metrics_result

if __name__ == "__main__":
    test_custom_metrics()