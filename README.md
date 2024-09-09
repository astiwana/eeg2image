# Mind Reader

My team and I developed this project during NatHacks 2023, where we earned 3rd place in the Research track.

## Introduction

Mind Reader aims to use live EEG recordings to predict the object a user is imagining. We built a GUI tool using PyQt and BrainFlow to collect training data. In total, we gathered 500 two-second samples, where a user focuses on a provided image while an OpenBCI Ganglion (4-channel EEG) records their brain activity. After detrending and applying a bandpass filter, we used wavelet decomposition to convert the EEG data into frequency-amplitude representations.

This processed data is fed into a transformer model built with PyTorch and CLIP, which classifies the object the user is imagining or viewing. So far, we've achieved 21% accuracy on a 10-class problem—2.1 times better than random guessing.

In the future, we hope our GUI tool can be adapted by others to collect their own training data and develop models. We also anticipate improved results with more EEG channels and larger training datasets.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
