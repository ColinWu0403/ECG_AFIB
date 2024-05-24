# ECG Signal Analysis and Atrial Fibrillation (AFIB) detection

### About
This project is a classification model in Python that predicts if a patient has Atrial Fibrillation (AFIB) based on the given ECG (Electrocardiography) signals of the patient.

The model is written in Python using scikit-learn for classification and uses wfdb and Neurokit2 to analyze ECG signals and measure data to be stored in .csv files

The ECG signals are taken from the MIT-BIH Atrial Fibrillation Database. The afdb folder contains the signals, but the actual files are not pushed as they are too large. Download them on the official website.

### Install required libraries
```
install --no-cache-dir -r requirements.txt
```

### Citations
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. https://doi.org/10.13026/C2MW2D
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
- Xie, C., McCullum, L., Johnson, A., Pollard, T., Gow, B., & Moody, B. (2023). Waveform Database Software Package (WFDB) for Python (version 4.1.0). PhysioNet. https://doi.org/10.13026/9njx-6322.