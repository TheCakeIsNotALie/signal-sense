from collections import defaultdict
from functools import reduce
from pathlib import Path
from dataclasses import dataclass
import struct 
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from copy import deepcopy
import os
from typing_extensions import Self

class ByteReader:
    i:int = 0
    data: bytes
    
    def __init__(self, data : bytes) -> None:
        self.data = data
    
    def read_bytes(self, bytes_to_read:int) -> bytes:
        ret_data = self.data[self.i: self.i + bytes_to_read]
        self.i += bytes_to_read
        return ret_data

@dataclass
class Dataset:
    data: list[list[float]]
    truth: list[list[int]]
    
    def get_truth_pe_distribution(self) -> dict:
        print(self.truth.shape)
        # print(np.sum(truth_list, axis=1).shape)
        uniques, counts = np.unique(np.sum(self.truth, axis=1), return_counts=True)
        return dict(zip(uniques, counts))
    
    def shuffle(self):
        dataset = list(zip(self.data, self.truth))
        # shuffle the dataset
        np.random.shuffle(dataset)
        # apply the shuffled state 
        self.data, self.truth = zip(*dataset)
        
    
    def save_to_file(self, filePath: str):
        np.savez(filePath, data=self.data, truth=self.truth)
        
    def __str__(self) -> str:
        return f"Dataset (data : {self.data.shape}, truth : {self.truth.shape})"

@dataclass
class SimEvent:
    event_id: int
    energy: float
    n_pe: int
    n_pixels: int 
    nn_PMT_channels: int 
    nn_fadc_point: int 
    fadc_sample_in_ns: float
    NGB_rate_in_MHz: float
    fadc_electronic_noise_RMS: float
    channel_waveforms: list[list[int]]
    channel_pe_truths: list[list[int]]
    
    def stitched_waveforms(self) -> tuple[list[int], list[int]] :
        """
        Tuple of (signal, pe_truth) from a simulation event
        """
        return (np.array(self.channel_waveforms).flatten(), np.array(self.channel_pe_truths).flatten())

    def waveform_windows(self, window_size: int) -> tuple[list[list[list[int]]], list[list[list[int]]]] :
        """
        Tuple of (signal, pe_truth) from a simulation event divided in sliding windows
        """
        signals = np.apply_along_axis(sliding_window_view, 1, arr=self.channel_waveforms, window_shape=window_size)
        truths = np.apply_along_axis(sliding_window_view, 1, arr=self.channel_pe_truths, window_shape=window_size)
        
        return (signals, truths)

    def training_waveform_windows(self, window_size: int, max_dataset_count: int = 500) -> Dataset :
        """
        Tuple of (Training Dataset, Validation Dataset) from a simulation event.
        Each variable is a list of windows.
        """
        
        sig, truth = self.waveform_windows(window_size)
        
        # shape sig and truth in [x, window_size] list form and zip them together
        dataset = list(zip(deepcopy(sig.reshape(-1, sig.shape[-1])), deepcopy(truth.reshape(-1, truth.shape[-1]))))
        # print(len(dataset))
        # shuffle the dataset to avoid getting the begining of the signal only
        np.random.shuffle(dataset)
        
        # prune dataset to avoid having almost only 0 in signal
        counts = dict()
        def filter_arr(entry : tuple[list[int], list[int]]):
            nonlocal counts
            
            pe_count = np.sum(entry[1])
            
            if counts.get(pe_count) is None:
                # print(f"Found pe_count {pe_count}")
                counts[pe_count] = 1
            elif counts.get(pe_count) < max_dataset_count:
                counts[pe_count] += 1
                
            return counts[pe_count] < max_dataset_count

        filtered_dataset = list(filter(filter_arr, dataset))
        # print(counts)
        
        data, truth = zip(*filtered_dataset)
        
        return Dataset(np.asarray(data), np.asarray(truth))
    
    def __str__(self) -> str:
        return f"event_id : {self.event_id}\n" + \
                f"energy : {self.energy}\n" + \
                f"n_pe : {self.n_pe}\n" + \
                f"n_pixels : {self.n_pixels}\n" + \
                f"nn_PMT_channels : {self.nn_PMT_channels}\n" + \
                f"nn_fadc_point : {self.nn_fadc_point}\n" + \
                f"fadc_sample_in_ns : {self.fadc_sample_in_ns}\n" + \
                f"NGB_rate_in_MHz : {self.NGB_rate_in_MHz}\n" + \
                f"fadc_electronic_noise_RMS : {self.fadc_electronic_noise_RMS}"

def dataset_split(self, ratio: float = 0.8) -> tuple[Dataset, Dataset]:
    """
    Split the dataset in two with a given ratio.
    """
    dataset = list(zip(self.data, self.truth))
    # split in training + validation
    train_size = int(len(dataset) * ratio)
    train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
        
    train_data, train_truth = zip(*train_dataset)
    train_data, train_truth = np.asarray(train_data), np.asarray(train_truth)
    val_data, val_truth = zip(*val_dataset)
    val_data, val_truth = np.asarray(val_data), np.asarray(val_truth)
    
    return (Dataset(train_data, train_truth), Dataset(val_data, val_truth))

def load_dataset_from_file(filePath: str) -> Dataset:
    loaded_data = np.load(filePath)
    return Dataset(loaded_data['data'], loaded_data['truth'])

def dataset_from_simulations(sim_events: list[SimEvent], window_size: int, max_dataset_count: int = 100) -> Dataset:
    all_sets = [ev.training_waveform_windows(window_size, max_dataset_count) for ev in sim_events]
    
    data_arr = np.vstack(list(map(lambda set: set.data, all_sets)))
    truth_arr = np.vstack(list(map(lambda set: set.truth, all_sets)))
    
    return Dataset(data_arr, truth_arr)
    
    # final_set = (np.array([]).reshape(0,window_size),np.array([]).reshape(0,window_size),np.array([]).reshape(0,window_size),np.array([]).reshape(0,window_size))
    # for set in all_sets:
    #     final_set = (np.vstack([final_set[0], set[0]]),np.vstack([final_set[1], set[1]]),np.vstack([final_set[2], set[2]]),np.vstack([final_set[3], set[3]]))
    
    # return final_set

def simevents_from_dir(dirPath: str) -> list[SimEvent] :
    """
    Read all binary output files in given folder from the corsika simulations.
    """
    events = list()
    
    for file in os.listdir(dirPath):
        if file.endswith(".bin"):
            events.append(simevent_from_file(os.path.join(dirPath, file)))
    
    return events

def simevent_from_file(filePath: str) -> SimEvent : 
    """
    Read binary output file from the corsika simulations.
    From the binary data, reproduce the waveform + photon detection truth.
    """
    print(f"Parsing {filePath}")
    data = Path(filePath).read_bytes()
    br = ByteReader(data=data)
    
    event_id = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)
    energy = struct.unpack('f', br.read_bytes(4))[0]
    n_pe = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)
    n_pixels = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)
    nn_PMT_channels = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)
    nn_fadc_point = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)
    fadc_sample_in_ns = struct.unpack('f', br.read_bytes(4))[0]
    NGB_rate_in_MHz = struct.unpack('f', br.read_bytes(4))[0]
    fadc_electronic_noise_RMS = struct.unpack('f', br.read_bytes(4))[0]
    
    channel_waveforms = []
    channel_pe_truths = []
    for c in range(0, nn_PMT_channels):
        channel_waveforms.append([])
        channel_pe_truths.append([])
        for t in range(0, nn_fadc_point):
            channel_waveforms[c].append(float(int.from_bytes(br.read_bytes(4), byteorder="little", signed=False)))
            channel_pe_truths[c].append(0)
    
    # print(len(channel_pe_truths))
    # print(fadc_sample_in_ns)
    while br.i < len(br.data) :
        ch_id = int.from_bytes(br.read_bytes(4), byteorder="little", signed=False) - 1
        # print(f"byte {br.i}  : ch_id {ch_id}")
        pe_hit_time = struct.unpack('f', br.read_bytes(4))[0]
        
        sample_delay = 2
        pe_hit_sample = math.ceil(pe_hit_time / fadc_sample_in_ns) + sample_delay
        if(pe_hit_sample >= 0 and pe_hit_sample < nn_fadc_point):
            channel_pe_truths[ch_id][pe_hit_sample] += 1
        
    return SimEvent(
        event_id = event_id,
        energy = energy,
        n_pe = n_pe,
        n_pixels = n_pixels,
        nn_PMT_channels = nn_PMT_channels,
        nn_fadc_point = nn_fadc_point,
        fadc_sample_in_ns = fadc_sample_in_ns,
        NGB_rate_in_MHz = NGB_rate_in_MHz,
        fadc_electronic_noise_RMS = fadc_electronic_noise_RMS,
        channel_waveforms = channel_waveforms,
        channel_pe_truths = channel_pe_truths,
    )
    
    
if __name__ == "__main__":
    # create data set and save to file 
    # sevents = simevents_from_dir("./data/bin_data")
    # print("parsing done")
    # dataset = dataset_from_simulations(sevents, window_size=21)
    # print("windowing done")
    # dataset.data /= 8.25
    # print(dataset.get_truth_pe_distribution())
    # dataset.save_to_file("./data/gamma_distributed_delayed.cache.npz")
    
    # load dataset from file
    dataset = load_dataset_from_file("./data/gamma_distributed_delayed.cache.npz")
    dataset.shuffle()
    # show all counts of dataset
    np.set_printoptions(threshold=np.inf)
    print(f"{np.sum(dataset.truth, axis=1)}") 
    
    # --------------------------------
    # Moving window Dataset display
    # --------------------------------
    # i = 239474
    # fig = plt.figure(figsize=(12, 5))
    
    # x = np.argwhere(np.sum(dataset.truth, axis=1) > 400)
    # print(f"event where pe > 0 {x}")

    # def update():
    #     plt.clf()
    #     ax1 = plt.subplot(211)
    #     ax1.plot(dataset.data[i], label='Sensor data')
    #     ax1.set_ylabel('Amplitude')
    #     ax2 = plt.subplot(212, sharex=ax1)
    #     ax2.set_ylabel('Number of pe events')
    #     ax2.plot(dataset.truth[i], label='Truth')
    #     plt.xlabel('Sample index')
    #     plt.draw()

    # def press(event):
    #     global i, ax1, ax2
    #     # Move window left or right
    #     if event.key == 'left':
    #         i-=1
    #     elif event.key == 'right':
    #         i+=1
    #     update()

    # fig.canvas.mpl_connect('key_press_event', press)
    # update()
    # plt.show()
    
    # print(train_data.shape, train_truth.shape, val_data.shape, val_truth.shape)
    # sevent = simevent_from_file("./data/gamma_ev0_out.bin")
    # print(sevent.training_waveform_windows(window_size=21).get_truth_pe_distribution())
    # sevent = simevent_from_file("./data/gamma_ev78_out.bin")
    # print(sevent.training_waveform_windows(window_size=21).get_truth_pe_distribution())
    # sevent = simevent_from_file("./data/gamma_ev89000_out.bin")
    # print(sevent.training_waveform_windows(window_size=21).get_truth_pe_distribution())
    # sevent = simevent_from_file("./data/gamma_ev898128_out.bin")
    # print(sevent.training_waveform_windows(window_size=21).get_truth_pe_distribution())
    
    # sevent = simevent_from_file("./data/gamma_on_nsb_1x_ev78_out.bin")
    # signal, truth = sevent.stitched_waveforms()
    # sevent.training_waveform_windows(21)
    
    # --------------------------------
    # Channel by channel display
    # --------------------------------
    # i = 4932
    # resultsYMax = np.max(sevent.channel_pe_truths) + 0.1
    # fig = plt.figure(figsize=(12, 5))

    # def update():
    #     plt.clf()
    #     ax1 = plt.subplot(211)
    #     ax1.plot(sevent.channel_waveforms[i], label='Sensor data')
    #     ax1.set_ylabel('Amplitude')
    #     ax2 = plt.subplot(212, sharex=ax1)
    #     # ax2.set_ylim(-1,resultsYMax)
    #     ax2.set_ylabel('Number of pe events')
    #     ax2.plot(sevent.channel_pe_truths[i], label='Truth')
    #     # [None, ] is for the batch array dimension, since it's a single event, just add none
    #     # ax2.plot(model.predict(val_data[i][None,], verbose=0)[0], '--r', label='Prediction')
    #     plt.xlabel(f"Sample index - {i}")
    #     plt.draw()

    # def press(event):
    #     global i, ax1, ax2
    #     # Move window left or right
    #     if event.key == 'left':
    #         i-=1
    #     elif event.key == 'right':
    #         i+=1
    #     update()

    # fig.canvas.mpl_connect('key_press_event', press)
    # update()
    # plt.show()
    
    # --------------------------------
    # Channel sliding window display (for verification) 
    # --------------------------------
    # i = 4932
    # j = 0
    # fig = plt.figure(figsize=(12, 8))

    # window_size = 21
    # sig_win, truth_win = sevent.waveform_windows(window_size)

    # def update():
    #     plt.clf()
        
    #     win_x = np.arange(j,j+window_size)
        
    #     ax1 = plt.subplot(211)
    #     ax1.plot(sevent.channel_waveforms[i], label='Sensor data')
    #     ax1.plot(win_x, sig_win[i][j], label='Windowed sensor data')
    #     ax1.set_ylabel('Amplitude')
        
    #     ax2 = plt.subplot(212, sharex=ax1)
    #     ax2.plot(sevent.channel_pe_truths[i], label='Truth')
    #     ax2.plot(win_x, truth_win[i][j], label='Windowed truth')
    #     ax2.set_ylabel('Number of pe events')
        
    #     plt.xlabel(f"Sample index - Channel {i} - Window {j}")
    #     plt.draw()

    # def press(event):
    #     global i, j
    #     # Move window left or right
    #     if event.key == 'left':
    #         j-=1
    #     elif event.key == 'right':
    #         j+=1
    #     elif event.key == 'up':
    #         i-=1
    #         j=0
    #     elif event.key == 'down':
    #         i+=1
    #         j=0
    #     update()

    # fig.canvas.mpl_connect('key_press_event', press)
    # update()
    # plt.show()
    
    
    # --------------------------------
    # Whole waveform display
    # --------------------------------

    # fig = plt.figure(figsize=(12, 5))
    # fig.suptitle('Waveform PE event prediction')
    # ax1 = plt.subplot(211)
    # ax1.plot(signal, label='Sensor data')
    # ax1.set_ylabel('Amplitude')
    # ax1.legend()
    # ax2 = plt.subplot(212, sharex=ax1)
    # ax2.set_ylabel('Number of pe events')
    # ax2.plot(truth, label='Truth')
    # # results = model.predict(sliced_data)
    # # avg_results = signal_sense.reaggregate_windows_avg(results, window_size, window_size-1)
    # # ax2.plot(avg_results, label='Prediction')
    # ax2.legend()
    # plt.xlabel('Sample index')

    # plt.show()