# ECG Signal Analysis and Atrial Fibrillation (Afib) Detection Model

[![Static Badge](https://img.shields.io/badge/Python-3.11.7-306998)](https://www.python.org/downloads/release/python-3117/)

## About
This project is a classification model in Python that predicts if a patient has Atrial Fibrillation (Afib) based on the given Electrocardiography (ECG) signals of the patient.

The model is written in Python using scikit-learn and TensorFlow Keras for classification and uses wfdb and Neurokit2 to analyze ECG signals and measure data to export them to a dataset (in the data/ folder as .csv files).

The ECG signals are taken from the MIT-BIH Atrial Fibrillation Database (afdb) and the PTB-XL ECG Database (ptb).

The afdb/ and ptb/ folders contain the respective database signals, but the actual files are not pushed as they are too large. You can download them on the official website (the link is in the citation below).

## Install required libraries
```
install --no-cache-dir -r requirements.txt
```

## Models

### Random Forest Classifier
Used sci-kit learn's RandomForestClassifier to classify an ECG signal in a small-time interval as Normal or AFIB.

### LSTM
Used TensorFlow to create a 3-layer LSTM (RNN) model to classify ECG signals in a 30-second time interval as Normal or AFIB.

### CNN
Used TensorFlow to create a 3 layer CNN model to classify ECG signals in a 30-second time interval as Normal or AFIB.

### SVM
Used sci-kit learn's SVC model to classify ECG signals in a 30-second time interval as Normal or AFIB.

### Gradient Boost
Used xgboost to create a Gradient Boosting model to classify ECG signals in a 30-second time interval as Normal or AFIB.

### Resnet
Used TensorFlow to create a model with 1 Convolutional layer and 2 Residual blocks to classify ECG signals in a 30-second time interval as Normal or AFIB.

## Citations
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. https://doi.org/10.13026/C2MW2D
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45.
- Xie, C., McCullum, L., Johnson, A., Pollard, T., Gow, B., & Moody, B. (2023). Waveform Database Software Package (WFDB) for Python (version 4.1.0). PhysioNet. https://doi.org/10.13026/9njx-6322.
