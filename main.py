import sys
import os
import json
import subprocess
import glob
import multiprocessing
from DenseTransformer import DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

config = json.load(open('./config.json'))

def get_text_from_image_path(image_path): 
	'''Runs the installed Tesseract OCR cli on the image at image_path and 
	saves the results in a text file in a folder sibling to the path.'''
	
	# grab the path data
	path, image = os.path.split(image_path)
	output_path = '{0}/processed'.format(path)
	
	# if the image has not already ben parsed, parse it
	os.makedirs(output_path, exist_ok=True)
	if not os.path.exists('{0}/{1}.txt'.format(output_path, image)):
		subprocess.call('tesseract {0} {1}/{2} -l jpn'.format(image_path, output_path, image))

	# return the image text
	return open('{0}/{1}.txt'.format(output_path, image), encoding='utf-8').read()

def generate_data_from_config(data_config):
	'''Generates the x (data) and y (target) data to use in the classifier 
	from a data configuration of the form [image_glob, classification].'''
	
	# initialize arrays
	data = []
	target = []

	# create a pool of threads for processing the images; leave one core vacant
	# so you can do something else while you wait for the results to finish
	pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
	
	# for each classificatoin
	for image_glob, classification in data_config:

		# parse out the image paths
		image_paths = glob.glob(image_glob)

		# map each image to a thread running get_vector_from_image_path,
		# add the results to the data array, and add classification to the
		# target array for each image in the image glob
		# data.extend(pool.map(get_vector_from_image_path, image_paths)) # DISABLED to use CountVectorizer
		data.extend(pool.map(get_text_from_image_path, image_paths))
		target.extend([classification] * len(image_paths))

	return (data, target)

def create_classifier(training_data_config):
	'''Generates a new classifier using the given training_data_config, which
	is an array of arrays of the form [image_glob, classification].'''

	# generate the data from the config
	data, target = generate_data_from_config(training_data_config)

	# build the vectorizer
	vectorizer = CountVectorizer(
		min_df=1, 
		ngram_range=(config['min_n_gram'],config['max_n_gram']), 
		token_pattern=config['token_pattern'])

	# build the pipeline for the selected algorithm
	algorithm = config['algorithm']
	if algorithm == 'LinearSVC':
		classifier = Pipeline([
			('vectorizer', vectorizer),
			('classifier', LinearSVC(C=config['penalty'], tol=config['tolerance']))
		])
	elif algorithm == 'SGD':
		classifier = Pipeline([
			('vectorizer', vectorizer),
			('classifier', SGDClassifier())
		])
	elif algorithm == 'RandomForest':
		classifier = Pipeline([
			('vectorizer', vectorizer),
			('classifier', RandomForestClassifier())
		])
	elif algorithm == 'GaussianNB':
		classifier = Pipeline([
			('vectorizer', vectorizer),
			('densifier', DenseTransformer()), 
			('classifier', GaussianNB())
		])
	else:
		sys.exit('Invalid algorithm selected: {0}'.format(algorithm))
	
	classifier.fit(data, target)

	return classifier

def test_classifier(classifier, test_data_config):
	'''Test a classifier using the given test_data_config, which is an array
	of arrays of the form [image_glob, classification].'''

	# generate the data from the config
	data, target = generate_data_from_config(test_data_config)

	# score and return
	return classifier.score(data, target)
	# classifier.predict(data)

def classify_receipts(classifier, image_glob):
	'''Uses a classifier to classify an glob of images and returns the results.'''
	
	## TODO: Multithreading, might be low ROI, especially if the typical use case is small

	results = []

	# for each image in the glob
	for image_path in glob.glob(image_glob):

		# read the image
		image_text = get_text_from_image_path(image_path)

		# use predict to classify, and add to results
		results += [(image_path, classifier.predict([image_text])[0])]

	return results

if __name__ == '__main__': 

	# create a classifier from the configured training data
	classifier = create_classifier(config['training_data'])

	# check for a passed argument and classify if it exists
	if len(sys.argv) == 2:
		image_glob = sys.argv[1]
		for image_path, result in classify_receipts(classifier, image_glob):
			print('{0}: {1}'.format(image_path, result))

	# otherwise, run on the test set and output accuracy
	else:
		accuracy = test_classifier(classifier, config['test_data'])
		print('Classification accuracy: {:.2%}'.format(accuracy))