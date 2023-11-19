import sys
import numpy as np
import pickle
import requests
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations

# Load CIFAR-10 batch
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta_file = r'batches.meta'
meta_data = unpickle(meta_file)

# Load the CIFAR-10 dataset
batch = unpickle("data_batch_1")
data = batch[b'data']
labels = batch[b'labels']
label_names = meta_data[b'label_names']

class EEGManager:
    def __init__(self) -> None:
        params = BrainFlowInputParams()
        params.serial_port = "COM3"
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
        for channel in eeg_channels:
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], sampling_rate, 0.5, 55.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        return data

    def __del__(self):
        if self.board.is_prepared():
            self.board.release_session()

class DisplayImageWidget(QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)

        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_text = QLabel(self)
        self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label_image)
        layout.addWidget(self.label_text)

    def set_image_data(self, image_data, label_index):
        h, w = 32, 32
        rgb_reshaped = np.reshape(image_data, (h, w, 3), order='F')
        rotated_image = np.rot90(rgb_reshaped, 3)
        resized_image = cv2.resize(rotated_image, (400, 400), interpolation=cv2.INTER_LINEAR)

        # Qt expects 32bit BGRA data for color images:
        bgra = np.empty((400, 400, 4), np.uint8, 'C')
        bgra[..., 0] = resized_image[..., 2]
        bgra[..., 1] = resized_image[..., 1]
        bgra[..., 2] = resized_image[..., 0]
        bgra[..., 3].fill(255)

        result = QImage(bgra.data, 400, 400, QImage.Format.Format_ARGB32)
        result.ndarray = bgra

        pixmap = QPixmap.fromImage(result)

        self.label_image.setPixmap(pixmap)

        # Increase the font size
        font = self.label_text.font()
        font.setPointSize(50)  # Set the desired font size
        self.label_text.setFont(font)

        self.label_text.setText(f"{label_index}")

class ImageDisplayGUI(QMainWindow):
    def __init__(self):
        super(ImageDisplayGUI, self).__init__()
        
        self.eeg_manager = EEGManager()
        self.eeg_manager.start()
        self.dataset_image = []
        self.dataset_label = []
        self.dataset_eeg = []

        self.label_index = QLabel(self)
        self.label_index.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.label_index.setStyleSheet("background-color: white; padding: 5px;")  # Optional styling

        self.setWindowTitle("EEG to Image")
        self.resize(1200, 800)
        icon = QIcon('logo.png')
        self.setWindowIcon(icon)

        self.central_widget = QWidget()
        self.main_layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.image_widget = DisplayImageWidget()

        self.main_layout.addWidget(self.image_widget, 0, 0)

        self.current_image_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_next_image)
        self.timer.start(2000)  # Set the timer interval in milliseconds (2000 ms = 2 seconds)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_dataset)
        self.main_layout.addWidget(self.save_button, 2, 0)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_timer)
        self.resume_button = QPushButton("Resume", self)
        self.resume_button.clicked.connect(self.resume_timer)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.resume_button)

        self.main_layout.addLayout(button_layout, 1, 0)

        self.paused = False
        self.show_white_screen = False
        self.white_screen_timer = QTimer(self)
        self.white_screen_timer.timeout.connect(self.toggle_white_screen)

    def display_next_image(self):
        if not self.paused:
            if self.show_white_screen:
                white_image = QImage(400, 400, QImage.Format.Format_ARGB32)
                white_image.fill(Qt.GlobalColor.white)
                pixmap = QPixmap.fromImage(white_image)
                self.image_widget.label_image.setPixmap(pixmap)
                self.image_widget.label_text.clear()
                self.show_white_screen = False
                self.white_screen_timer.start(500)  # Set the white screen duration in milliseconds (500 ms = 0.5 seconds)
            else:
                if self.current_image_index < len(data):
                    self.current_image_index=np.random.randint(0,len(data))
                    image_data = data[self.current_image_index]
                    label_index = label_names[labels[self.current_image_index]]
                    label_index = label_index.decode('UTF-8')
                    self.image_widget.set_image_data(image_data, label_index)
                    self.label_index.setText(f"Count: {self.current_image_index}")
                    if self.current_image_index != 0:
                        image_data = data[self.current_image_index - 1]
                        rgb_reshaped = np.reshape(image_data, (32, 32, 3), order='F')
                        rotated_image = np.rot90(rgb_reshaped, 3)
                        label_index = labels[self.current_image_index - 1]
                        eeg_signal = self.eeg_manager.get_data(2)
                        self.dataset_image.append(rotated_image)
                        self.dataset_label.append(label_index)
                        self.dataset_eeg.append(eeg_signal)
                    self.show_white_screen = True
                    self.current_image_index += 1
                    self.save_dataset()
                    #self.current_image_index=len(data)
                else:
                    self.timer.stop()  # Stop the timer when all images are displayed

    def toggle_white_screen(self):
        self.white_screen_timer.stop()
        self.image_widget.label_image.clear()

    def pause_timer(self):
        self.paused = True

    def resume_timer(self):
        self.paused = False
        self.timer.start(2000)  # Restart the timer when resuming

    def save_dataset(self):

        eeg = np.asarray(self.dataset_eeg)
        #print(eeg.shape)  # (15, 400)
        self.paused = True
        #put your servers ip here:
        r = requests.post('http://127.0.0.1:3289/', json={'eeg': eeg.tolist()})
        print(r.json())
        self.dataset_eeg = []
        
        #np.save("data/dataset_image.npy", self.dataset_image)
        #np.save("data/dataset_label.npy", dataset_label)
        #np.save("datalive/dataset_eeg_0.npy", dataset_eeg)

def main():
    app = QApplication([])
    window = ImageDisplayGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
