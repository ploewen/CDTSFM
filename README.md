# Cross Domain Foundation Models

In this project I aim to compare the zero shot performance of Foundation Models for classification
tasks in different domains. To this end I will source data from economics, healthcare,
finance and engineering. I am to compare one model from each of these domains as well as 
some general foundation models, custom embedding models and a random embedding model.

## Data

### PTB-XL 
PTB-XL [1] is a large publicly available electrocardiography dataset [2]. The dataset contains 21799 ECG 10 second readings using 12 leads from 18869 patients. The data has been expert annotated, conforming to the SCP-ECG standard. Included annotations are "NORM" (Normal ECG), "MI" (Myocardial Infarction), "STTC" (ST/T Change), "CD" (Conduction Disturbance), and "HYP" (Hypertrophy).

The ECG recordings were recorded at a frequency of 500 Hz and downsampled to 100Hz. Recommended 10-fold train-test splits have been provided by the authors for ease of cross
validation training and testing.

The [database](https://physionet.org/content/ptb-xl/1.0.3/) is hosted on PhysioNet [3]. 

### ESC-50

ESC-50 [4] is a dataset containing 2000 environmental audio recordings. The recordings are uniformly sampled at 44.1kHz with a fixed length of 5 seconds.

Each recording is organized into one of 50 semantic classes, with 40 examples per class. The available classes can be categorized as animals, natural soundscapes, water sounds, human non-speech sounds, interior/domestic sounds, exterior/urban sounds [4].

The data is pre aranged by the author into 5 folds for cross validation. We reserve one of these splits to be our testing dataset, using the remaining 4 splits as testing. This ensures balanced training and testing sets.

A [GitHub repository](https://github.com/karolpiczak/ESC-50) containing the dataset and other information has been made available by the author.

### S&P 500 Constituent Returns

The S&P 500 (Standard and Poor's 500) is an index fund comprised of the 500 largest stocks. Our dataset is comprised of the daily returns of each constituent stock of the index, from January 1st 2000 to November 1st 2025. It has been sourced through the Yahoo Finance API [5] by usage of the yfinance python package.

To pre-process the data we adapt the methodology put forward in [6]. We do so by creating windows of 1000 days, reserving the first 749 days (approximately 3 trading years) of the window for our training set and the last 250 (approximately 1 trading year) days for our testing set. We have split the data in this manner to ensure there is no data leakage, adopting purging (no overlap between training and testing sets) and embargo (removing any observation directly following a training set) methods from [7].

Financial literature suggests that forecasting based on classifying the direction of the return generally performs better than level estimation [8]. To this end, we set the input variables of our models to be composed of the returns from sliding window of 20 days (approximately 1 trading month) with the output variable set as 1 or 0 if the following day's return is above or below the window's median return respectively.

## References
<ol>
  <li>
    P. Wagner, N. Strodthoff, R.-D. Bousseljot, W. Samek, and T. Schaeffter, “PTB-XL, a large publicly available electrocardiography dataset,” PhysioNet, Nov. 2022, doi: <a href="https://doi.org/10.13026/kfzx-aw45">https://doi.org/10.13026/kfzx-aw45</a>.
  </li>
  <li>
    P. Wagner et al., “PTB-XL, a large publicly available electrocardiography dataset,” Scientific Data, vol. 7, no. 1, May 2020, doi: <a href="https://doi.org/10.1038/s41597-020-0495-6">https://doi.org/10.1038/s41597-020-0495-6</a>.
  </li>
  <li>
    A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet,” Circulation, vol. 101, no. 23, June 2000, doi: <a href="https://doi.org/10.1161/01.cir.101.23.e215">10.1161/01.cir.101.23.e215</a>.
  </li>
  <li>
    K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” Proceedings of the 23rd ACM International Conference on Multimedia, pp. 1015–1018, 2015, doi: <a href="https://doi.org/10.1145/2733373.2806390">https://doi.org/10.1145/2733373.2806390</a>.
  </li>
  <li>
    Yahoo Finance, “Yahoo Finance - Business Finance, Stock Market, Quotes, News,” Yahoo Finance, 2025. <a href="https://finance.yahoo.com/">https://finance.yahoo.com/</a>
  </li>
  <li>
    C. Krauss, X. A. Do, and N. Huck, “Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500,” European Journal of Operational Research, vol. 259, no. 2, pp. 689–702, Jun. 2017, doi: <a href="https://doi.org/10.1016/j.ejor.2016.10.031">https://doi.org/10.1016/j.ejor.2016.10.031</a>.
  </li>
  <li>
    M. Lopez De Prado, Advances in financial machine learning. New Jersey: Wiley, 2018.
  </li>
  <li>
    M. T. Leung, H. Daouk, and A.-S. Chen, “Forecasting Stock Indices: A Comparison of Classification and Level Estimation Models,” SSRN Electronic Journal, 1999, doi: <a href="https://doi.org/10.2139/ssrn.200429">https://doi.org/10.2139/ssrn.200429</a>.
  </li>
</ol>