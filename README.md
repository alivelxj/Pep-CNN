# Pep-CNN
Pep-CNN: A method for predicting therapeutic peptides based on convolutional neural network

# Usage
  1.The datasets file contains AAP, ABP, ACP, AIP, AVP, CPP, QSP, SBP.  
  2.BE, EBGW, EGAAC, BLOSUM62, KNN are the implementation of Feature extraction.  
  3.SVM, RF, XGBoost, CNN, DNN, LSTM, imCNN are the implementation of classifier.  

  Configuration Environment：python=3.7, tensorflow=2.2.0, keras=2.4.3, numpy=1.19.5  

  How to use our model?  
  First, run complementary.py to make the dataset under the data file equal in length.  
  Second, extract features using the five feature extraction methods in the feature extraction file, and then run montage.py in the feature extraction file to combine them together.  
  Finally, run imCNN.py in the classifier folder to get the prediction results.  
