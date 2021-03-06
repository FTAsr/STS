## This file is the point of invocation for the ASAG system.

import sys
sys.path = ['../', '../feedbackmodels', '../featuremodels'] + sys.path
import models
from featuremodels import models as md
import pickle
import AllScores_supervised_02 as app
from properties import *
import nltk

def post_process(result):
	# Post processing model's predicted score as required.
	if GRADING_SCALE_REAL_0_1:
		result['prediction'] /= 5.0
	elif GRADING_SCALE_Integer_0_5:
		result['prediction'] = np.floor(result['prediction'])
	elif GRADING_SCALE_MULTICLASS:
		result.loc[(result.prediction >= 0) & (result.prediction < 1), 'prediction'] = GRADING_SCALE_LABELS[0]
		result.loc[(1 <= result.prediction) & (result.prediction < 4), 'prediction'] = GRADING_SCALE_LABELS[1]
		result.loc[(4 <= result.prediction) & (result.prediction <= 5), 'prediction'] = GRADING_SCALE_LABELS[2]
	return result

if __name__ == '__main__':

	classifier = pickle.load(open('../pretrained/classifiers/' + BEST_CLASSIFIER_COLLEGE, 'rb'))
	usedModels = list()
	#bowm = md.bow("../embeddings/GoogleNews-vectors-negative300.bin")
	feedm = md.feedback()
	fbm = md.featureBased()
	
	## comment the lines corressponding to unused models
	#usedModels.append(bowm)
	usedModels.append(feedm)
	usedModels.append(fbm)

	feedm.feedback_model.build_vocab_k_words(K=500000)

	#feedback_model = models.loadFeedbackModel()
	print '\nWelcome to Automatic Short Answer Grading system. \n'	
	

	while True:
		flag = raw_input('Enter Yes or Y to proceed and No or N to abort. \n')
		if flag.lower() == 'yes' or flag.lower() == 'y':
			try:
				if INTERACTIVE_MODE:
					goldA = raw_input('Enter gold Answer: \n')
					studA = raw_input('Enter students\'s Answer: \n')
					score = raw_input('Enter target score:')
					threshold = raw_input('Enter threshold for word importance:')
					input_scale = raw_input('Choose 1 for 0-1 scale, 2 for 0-5 scale:')
					if int(input_scale) == 1:
						score = float(score) * 5

					testSet = [[goldA], [studA], [float(score)]]
					result = app.test(usedModels, classifier, testSet)
					# Covert result into required grading scale.
					result = post_process(result)
					print result
					
					#feedback_model.build_vocab([sentA, sentB], tokenize=True)
					keywords = models.feedback(feedm.feedback_model, goldA, float(threshold))
					#print 'keywords:', keywords
					sent = nltk.word_tokenize(studA)
					sent = [word.lower() for word in sent]
					#print 'sent', sent
					feedback = set(keywords) - set(sent) - set(['<s>', '</s>'])
					print 'Hey there, you missed following keywords:\n', list(feedback)
					

				elif BATCH_MODE:
					input_path = raw_input('Enter the complete file path for test data. \n')
					trainSet, devSet, testSet = app.load_data_nosplit(input_path)
					result = app.test([feedm, fbm], classifier, testSet)
					result = post_process(result)
					timestamp = time.strftime("%Y%m%d-%H%M%S")
					result.to_csv('../output/result' + timestamp + '.csv')

			except Exception, e:
				print "Error: %s" % e
				print 'Please correct the error and try again.'

		else:
			break





