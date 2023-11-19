import numpy as np
import pickle

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

# Load CIFAR-10 batch
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load the CIFAR-10 dataset
batch = unpickle("data_batch_1")
data = batch[b'data'] # contains the images each row is one image with 3072 colums first 1024 are red values then next 1024 are green and final 1024 are blue values
labels = batch[b'labels'] #  a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
label_names = batch[b'label_names'] # a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

class EEGManager:
    def __init__(self) -> None:
        params = BrainFlowInputParams()
        self.board_id = BoardIds.GANGLION_BOARD

        self.board = BoardShim(self.board_id, params)

        self.board.prepare_session()

    def start(self):
        self.board.start_stream()

    def get_data(self, secs):
        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        n_samples = sampling_rate * secs
        data = self.board.get_current_board_data(n_samples)

        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        DataFilter.detrend(data[eeg_channels], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data[eeg_channels], sampling_rate, 0.5, 55.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        return data

    def __del__(self):
        if self.board.is_prepared():
            self.board.release_session()

result_array = []

for i in range(len(data)):
    image = data[i]
    eeg_manager = EEGManager()

    eeg_signal = eeg_manager.get_data(2)

    object_name = label_names[labels[i]]
    # Store the result
    result_array.append((labels[i], eeg_signal, object_name))
