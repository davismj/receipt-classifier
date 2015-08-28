# Receipt Classifier for MerryBiz
This is an example of a receipt classifier using the Gaussian Naive Bayes supervised learning algorithm. The application uses Tesseract OCR to scan receipt images for text, and uses the occurrence each unicode character code as a parameter in Gaussian Naive Bayes. 

# Usage 

## Training
To train the application, set the `training_data` value in the `config.json` to an array of `[glob, classification]` values. Glob is a glob of image files to parse, and classification is 1 if the glob is a set of taxi receipts, 0 otherwise.

## Testing Accuracy
To test the accuracy of the training, configure a test data set similar to the training data set above by setting the `test_data` value in the `config.json`.

## Classifying Receipts
To classify a receipt or set of receipts, call `main.py` with a file or glob.

# Limitations
- The Tesseract OCR doesn't do an effective job of parsing the receipts. In particular, the characters 車 and 番 almost always appear on taxi receipts. However, since they are often incorrectly parsed, this information can't be used effectively.
- The naive bayes algorithm isn't nearly as effective on documents as short as receipts.
- The method of characteriziing a reciept involves creating a 46000 length array and counting features by the unicode integer value of each character in the document. These arrays are therefore quite sparse and memory inefficient.

# Dependencies
- Tesseract OCR 3.2 or greater
- numpy
- scipy
- scikit-learn
