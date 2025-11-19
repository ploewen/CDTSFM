# Cross Domain Time Series Foundation Models

In this project I aim to compare the zero shot performance of Time Series Foundation Models (TSFM) for classification
tasks in different domains. To this end I will source data from healthcare,
finance and engineering. I am to compare TSFMs from each of these domains as well as 
some general foundation models, custom embedding models and a random embedding model.

## Data

### PTB-XL 
PTB-XL  is a large publicly available electrocardiography dataset [1],[2]. The dataset contains 21799 ECG 10 second readings using 12 leads from 18869 patients. The data has been expert annotated, conforming to the SCP-ECG standard. Included annotations are "NORM" (Normal ECG), "MI" (Myocardial Infarction), "STTC" (ST/T Change), "CD" (Conduction Disturbance), and "HYP" (Hypertrophy).

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

### StarEmbed

The StarEmbed dataset is a multi-band light curve dataset designed to benchmark astronomical and general TSFMs on astronomical time series data [9]. The dataset is composed of light curves selected from release 23 (DR23) of the ZTF astronomical survey [10]. Star class labels have been expertly labeled in the Catalina Surveys Periodic Variable Star Catalog (CSPVS) [11].

The dataset is split into training, validation and testing datasets. Time series for green and red light curves are given, along with time stamps formatted in Modified Julian Dates (MJD), and a string identifying the class of the star. Several other variables are included but will not be used for our analysis. 

The stars were first pre-classified using multivariate kernel density estimation [12], followed by a classification done by visual inspection to asses their final class [11]. In total there are 6 different star classes present in the dataset:  "CVn", "EW", "EA", "LPV", "RRab", "RRc", "RRd", and "RS". All classes are present in approximately equal proportion in all three splits, however, some classes  have many fewer observations than others, for example, LPV and EW classes which contain 360 and 27075 observations respectively [11].

## References
<ol>
  <li>
    P. Wagner, N. Strodthoff, R.-D. Bousseljot, W. Samek, and T. Schaeffter, “PTB-XL, a large publicly available electrocardiography dataset,” PhysioNet, Nov. 2022, doi: 10.13026/kfzx-aw45.
  </li>
  <li>
    P. Wagner, N. Strodthoff, R.-D. Bousseljot, W. Samek, and T. Schaeffter., “PTB-XL, a large publicly available electrocardiography dataset,” *Scientific Data*, vol. 7, no. 1, May 2020, doi: 10.1038/s41597-020-0495-6.
  </li>
  <li>
    A. L. Goldberger *et al*. “PhysioBank, PhysioToolkit, and PhysioNet,” *Circulation*, vol. 101, no. 23, June 2000, doi: 10.1161/01.cir.101.23.e215.
  </li>
  <li>
    K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” *Proceedings of the 23rd ACM International Conference on Multimedia*, pp. 1015–1018, 2015, doi: 10.1145/2733373.2806390  .
  </li>
  <li>
    Yahoo Finance, “Yahoo Finance - Business Finance, Stock Market, Quotes, News,” Yahoo Finance, 2025. <a href="https://finance.yahoo.com/">https://finance.yahoo.com/</a>
  </li>
  <li>
    C. Krauss, X. A. Do, and N. Huck, “Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500,” *European Journal of Operational Research*, vol. 259, no. 2, pp. 689–702, Jun. 2017, doi: 10.1016/j.ejor.2016.10.031.
  </li>
  <li>
    M. L. De Prado, *Advances in financial machine learning*. New Jersey: Wiley, 2018.
  </li>
  <li>
    M. T. Leung, H. Daouk, and A.-S. Chen, “Forecasting Stock Indices: A Comparison of Classification and Level Estimation Models,” *SSRN Electronic Journal*, 1999, doi: 10.2139/ssrn.200429.
  </li>
  <li>
    W. Li *et al*., "StarEmbed: Benchmarking Time Series Foundation models on astronomical observations of variable stars," arXiv:2510.06200 [astro-ph.SR], Oct. 2025. 
  </li>
  <li>
   E. C. Bellm *et al*., “The Zwicky Transient Facility: System Overview, performance, and first results,” *Publications of the Astronomical Society of the Pacific*, vol. 131, no. 995, p. 018002, Dec. 2018, doi: 10.1088/1538-3873/aaecbe.
  </li>
  <li>
   A. J. Drake *et al*., “THE CATALINA SURVEYS PERIODIC VARIABLE STAR CATALOG,” *The Astrophysical Journal Supplement Series*, vol. 213, no. 1, p. 9, Jun. 2014, doi: 10.1088/0067-0049/213/1/9.
  </li>
  <li>
   D. W. Scott, *Multivariate density estimation: Theory, Practice, and Visualization*. John Wiley & Sons, 2015.
  </li>


</ol>
