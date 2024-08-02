# ECG Signal Analysis and Atrial Fibrillation (AFib) Detection Models

[![Static Badge](https://img.shields.io/badge/Python-3.11.7-306998)](https://www.python.org/downloads/release/python-3117/)

## About

This project contains a collection of classification models in Python that predict if a patient has Atrial Fibrillation (AFib) based on the given Electrocardiography (ECG) signals of the patient.

The models are written in Python using `scikit-learn` and `TensorFlow Keras`. `wfdb` and `Neurokit2` are used to analyze ECG signals and measure data to export them to a dataset (in the [data](data) folder as .csv files).

The ECG signals are taken from the open-source MIT-BIH Atrial Fibrillation Database (in [data/afdb/](data/afdb) and the PTB-XL ECG Database (in [data/ptb/](data/ptb)).

The [afdb](data/afdb) and [ptb](data/ptb) folders contain the respective database signals, but the actual files are not pushed as they are too large. You can download them on the official website (the link is in the citation below).

The [models](models) folder contains the generated files for the models, also not pushed to GitHub.

The [reports](reports) folder contains the auto-generated report for each model, including the accuracy and confusion matrix.

## Install required libraries

```
install --no-cache-dir -r requirements.txt
```

## Models

### Random Forest Classifier

Used RandomForestClassifier from `sci-kit learn` to classify an ECG signal on a 10-second interval as Normal or AFIB.

### LSTM

Used `TensorFlow` to create a 3-layer LSTM (RNN) model to classify ECG signals on 10-second intervals as Normal or AFIB.

### CNN

Used `TensorFlow` to create a 3-layer CNN model to classify ECG signals on 10-second intervals as Normal or AFIB.

### SVM

Used SVC from `sci-kit learn` to classify ECG signals on 10-second time intervals as Normal or AFIB.

### Gradient Boost

Used `xgboost` to create a Gradient Boosting model to classify ECG signals on 10-second intervals as Normal or AFIB.

### Resnet

Used `TensorFlow` to create a Resnet model with 1 Convolutional layer and 2 Residual blocks to classify ECG signals on 10-second intervals as Normal or AFIB.

## References

- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. https://doi.org/10.13026/C2MW2D
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45.
- Xie, C., McCullum, L., Johnson, A., Pollard, T., Gow, B., & Moody, B. (2023). Waveform Database Software Package (WFDB) for Python (version 4.1.0). PhysioNet. https://doi.org/10.13026/9njx-6322.
