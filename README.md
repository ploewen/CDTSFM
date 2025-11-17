# Cross Domain Foundation Models

In this project I aim to compare the zero shot performance of Foundation Models for classification
tasks in different domains. To this end I will source data from economics, healthcare,
finance and engineering. I am to compare one model from each of these domains as well as 
some general foundation models, custom embedding models and a random embedding model.

## Data

### PTB-XL (Wagner et al., 2020)
PTB-XL is a "large publicly available electrocardiography dataset" (Wagner et al., 2022). 

This dataset contains 21799 ECG 10 second readings using 12 leads from 18869 patients. The data has been expert annotated, conforming to the SCP-ECG standard. Included annotations are "NORM" (Normal ECG), "MI" (Myocardial Infarction), "STTC" (ST/T Change), "CD" (Conduction Disturbance), and "HYP" (Hypertrophy).

The ECG recordings were recorded at a frequency of 500 Hz and downsampled to 100Hz. Recommended 10-fold train-test splits have been provided by the authors for ease of cross
validation training and testing.

The [database](https://physionet.org/content/ptb-xl/1.0.3/) is hosted on PhysioNet (Goldberg et al., 2000). 

### ESC-50 (Piczak, 2015)

ESC-50 is  dataset containing 2000 environmental audio recordings. Each recording is organized into one of 50 semantic classes, with 40 examples per class. 

The data is pre aranged into 5 folds for testing and training. The recordings are uniformly sampled at 44.1kHz with a fixed length of 5 seconds making the dataset a good candidate for benchmarking.

A [GitHub repository](https://github.com/karolpiczak/ESC-50) containing the dataset and other information has been made available by the author.


## References

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset. *PhysioNet*. https://doi.org/10.13026/kfzx-aw45

‌Wagner, Patrick, et al. “PTB-XL, a Large Publicly Available Electrocardiography Dataset.” *Scientific Data*, vol. 7, no. 1, 25 May 2020, https://doi.org/10.1038/s41597-020-0495-6.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.