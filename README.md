# Receipt Classifier for MerryBiz
This is an example of a receipt classifier using the LinearSVC supervised learning algorithm. The application uses Tesseract OCR to scan receipt images for text, and uses the occurrence each unicode character code as a parameter in LinearSVC.

# Usage 

## Configuration
- algorithm: the algorithm to use, one of LinearSVM, GaussianNB, RandomForest, SGD
- penalty: the penalty parameter passed to LinearSVM, which reduces overfitting as it decreases
- tolerance: the algorithm stopping tolerance passed to LinearSVM, which trades computation time for better accuracy as it approaches 0
- min_n_gram / max_n_gram: the min / max n to use in counting features, which treats all n consecutive characters as a feature for min_n_gram <= n <= max_n_gram
- token_pattern: the regex to use in selecting words, defaults to single characters

## Training
To train the application, set the `training_data` value in the `config.json` to an array of `[glob, classification]` values. Glob is a glob of image files to parse, and classification is 1 if the glob is a set of taxi receipts, 0 otherwise.

## Testing Accuracy
To test the accuracy of the training, configure a test data set similar to the training data set above by setting the `test_data` value in the `config.json`.

## Classifying Receipts
To classify a receipt or set of receipts, call `main.py` with a file or glob.

# Dependencies
- Tesseract OCR 3.2 or greater
- numpy
- scipy
- scikit-learn

# Notes

This is technically an implementation of both n-gram and bag of words. Both n in the n-gram algorithm and the word regex token can be configured in the config.json. The default is n = 1, word = /\S/.

The application also supports several other algorithms in the scikit learn module: Gaussian Naive-Bayes, Stochastic Gradient Descent, Random Forest. However, these were added mostly for quick testing. They have not been configured or optimized, and the results were vastly inferior to LinearSVM, as expected.