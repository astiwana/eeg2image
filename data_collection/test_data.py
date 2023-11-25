import numpy as np
from brainflow.board_shim import BoardIds, BoardShim

print(BoardShim.get_sampling_rate(BoardIds.GANGLION_BOARD))

# print(np.load("data/dataset_eeg.npy"))
# print(np.load("data/dataset_image.npy"))
# print(np.load("data/dataset_label.npy"))