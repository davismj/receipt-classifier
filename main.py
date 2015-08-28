import sys
import os
import json
import subprocess
import glob
import multiprocessing
from sklearn.naive_bayes import GaussianNB

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

def get_vector_from_image_path(image_path):
	'''Parse the image into a vector where the dimension is the ord(char) 
	and the magnitude is the repeititon of the character in the text.'''
	
	# initialize the vector
	vector = [0] * 46000

	# for each character, increment the magnitude in that dimension by 1
	for char in get_text_from_image_path(image_path):
		vector[ord(char)] += 1 

	return vector

def generate_data_from_config(config):
	'''Generates the x (data) and y (target) data to use in the GaussianNB
	classifier from a data configuration of the form [image_glob, classification].'''
	
	# initialize arrays
	data = []
	target = []

	# create a pool of threads for processing the images; leave one core vacant
	# so you can do something else while you wait for the results to finish
	pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
	
	# for each classificatoin
	for image_glob, classification in config:

		# parse out the image paths
		image_paths = glob.glob(image_glob)

		# map each image to a thread running get_vector_from_image_path,
		# add the results to the data array, and add classification to the
		# target array for each image in the image glob
		data.extend(pool.map(get_vector_from_image_path, image_paths))
		target.extend([classification] * len(image_paths))

	return (data, target)

def create_classifier(training_data_config):
	'''Generates a new classifier using the given training_data_config, which
	is an array of arrays of the form [image_glob, classification].'''

	# generate the data from the config
	data, target = generate_data_from_config(training_data_config)

	# chunk_size is used to break apart the data into chunks to prevent overflow
	chunk_size = config['chunk_size']

	# create GaussianNB and train chunk-by-chunk using partial_fit
	classifier = GaussianNB()
	for n in range(len(data) // chunk_size):
		data_chunk = data[n*chunk_size:(n+1)*chunk_size]
		target_chunk = target[n*chunk_size:(n+1)*chunk_size]
		classifier.partial_fit(data_chunk, target_chunk, [0,1])

	return classifier

def test_classifier(classifier, test_data_config):
	'''Test a classifier using the given test_data_config, which is an array
	of arrays of the form [image_glob, classification].'''

	# generate the data from the config
	data, target = generate_data_from_config(test_data_config)

	# score and return
	return classifier.score(data, target)

def classify_receipts(classifier, image_glob):
	'''Uses a classifier to classify an glob of images and returns the results.'''
	
	results = []

	# for each image in the glob
	for image_path in glob.glob(image_glob):

		# read the image
		vector = get_vector_from_image_path(image_path)

		# use predict to classify, and add to results
		results += [(image_path, classifier.predict([vector])[0])]

	return results

if __name__ == '__main__': 

	# create a classifier from the configured training data
	classifier = create_classifier(config['training_data'])

	# check for a passed argument and classify if it exists
	if len(sys.argv) == 2:
		image_glob = sys.argv[1]
		for image_path, result in classify_receipts(classifier, image_glob):
			print('Classification result: {0}'.format(result))

	# otherwise, run on the test set and output accuracy
	else:
		accuracy = test_classifier(classifier, config['test_data'])
		print('Classification accuracy: {:.2%}'.format(accuracy))