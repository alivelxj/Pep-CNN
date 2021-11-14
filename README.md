# Pep-CNN
Pep-CNN: A method for predicting therapeutic peptides based on convolutional neural network

# Usage
1.The datasets file contains AAP, ABP, ACP, AIP, AVP, CPP, QSP, SBP.
2.BE, EBGW, EGAAC, BLOSUM62, KNN are the implementation of Feature extraction.
3.RF, XGboost, CNN, DNN, imCNN are the implementation of classifier.


How to use our model?
First, run complementary.py to make the dataset under the data file equal in length.
second, extract features using the five feature extraction methods in the feature extraction file, and then combine them together。
Finally, run imCNN.py in the classifier folder to get the prediction results.
