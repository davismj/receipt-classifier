import sys
import os
import json
import subprocess
import glob
import multiprocessing
from sklearn.naive_bayes import GaussianNB

config = json.load(open('./config.json'))

def get_text_from_image_path(image_path): 
	'''Runs the installed tesseract cli on the image at image_path.'''
	path, image = os.path.split(image_path)
	output_path = '{0}/processed'.format(path)
	os.makedirs(output_path, exist_ok=True)
	if not os.path.exists('{0}/{1}.txt'.format(output_path, image)):
		subprocess.call('tesseract {0} {1}/{2} -l jpn'.format(image_path, output_path, image))
	image_text = open('{0}/{1}.txt'.format(output_path, image), encoding='utf-8').read()
	return image_text

def get_vector_from_image_path(image_path):
	'''parse the image into a vector where the dimension is the ord(char) 
	and the magnitude is the repeititon of the character in the text'''
	vector = [0] * 46000
	image_text = get_text_from_image_path(image_path)
	for char in image_text:
		vector[ord(char)] += 1 
	return vector

def generate_data_from_config(config):
	data = []
	target = []
	pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
	
	# for each training data point
	for image_glob, classification in config:

		image_paths = glob.glob(image_glob)
		data.extend(pool.map(get_vector_from_image_path, image_paths))
		target.extend([classification] * len(image_paths))

	return (data, target)

def create_taxi_classifier(training_data_config):

	data, target = generate_data_from_config(training_data_config)
	chunk_size = config['chunk_size']

	# create GaussianNB and train
	classifier = GaussianNB()
	for n in range(len(data) // chunk_size):
		data_chunk = data[n*chunk_size:(n+1)*chunk_size]
		target_chunk = target[n*chunk_size:(n+1)*chunk_size]
		classifier.partial_fit(data_chunk, target_chunk, [0,1])
	return classifier

def test_taxi_classifier(classifier, test_data_config):
	data, target = generate_data_from_config(test_data_config)
	return classifier.score(data, target)

def classify_receipts(classifier, image_glob):
	
	results = []

	# get glob
	for image_path in glob.glob(image_glob):
		vector = get_vector_from_image_path(image_path)
		results += [(image_path, classifier.predict([vector])[0])]

	return results

if __name__ == '__main__': 

	classifier = create_taxi_classifier(config['training_data'])

	# get the passed image path
	if len(sys.argv) == 2:
		image_glob = sys.argv[1]
		for image_path, result in classify_receipts(classifier, image_glob):
			print(result)

	# otherwise, run on the test set and output accuracy
	else:
		print(test_taxi_classifier(classifier, config['test_data']))