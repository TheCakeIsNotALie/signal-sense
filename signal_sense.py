import tensorflow as tf
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

# def reaggregate_windows_avg(windows, window_size, overlap):
#     # Determine the step size between windows
#     step_size = window_size - overlap
    
#     # Calculate the length of the final signal
#     total_length = (len(windows) - 1) * step_size + window_size
    
#     # Initialize arrays to accumulate values and counts
#     accumulated_values = np.zeros(total_length)
#     count_contributions = np.zeros(total_length)
    
#     # Accumulate values and counts
#     for i, window in enumerate(windows):
#         start = i * step_size
#         end = start + window_size
#         accumulated_values[start:end] += window
#         count_contributions[start:end] += 1
    
#     # Average the accumulated values by counts
#     reaggregated_signal = accumulated_values / count_contributions
    
#     return reaggregated_signal

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

def run_tests(model : tf.keras.Model, window_size: int):
    configs = {
        "pe_rate_mhz": [20,50,64,150], # 1 to 150 MHz [20,32,50,64,128,150]
        "noise_lsb": np.arange(0, 1.5, 0.25), # 0 to 1.5 noise
    }
    
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