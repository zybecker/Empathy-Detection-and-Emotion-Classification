#!/usr/bin/env python
# Author: roman.klinger@ims.uni-stuttgart.de
# Evaluation script for Empathy shared task at WASSA 2024
# Adapted for CodaLab purposes by Orphee (orphee.declercq@ugent.be) in May 2018
# Adapted for multiple subtasks by Valentin Barriere in December 2021 (python 3), then in February 2022
# Adapted for multiple subtasks by Salvatore Giorgi in March 2024

from __future__ import print_function
import sys
import os
from math import sqrt
import csv

to_round = 4

nb_labels_CONVD = 1
nb_labels_CONVT = 3
nb_labels_EMP   = 2
nb_labels_PER   = 5

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def readFileToList(filename):
	#eprint("Reading data from",filename)
	lines=filename.readlines()
	result=[]
	for x in lines:
		result.append(x.rstrip().split('\t'))
	filename.close()
	return result

def readCSVToList(filename):
	#eprint("Reading data from",filename)
	with open(filename.name, newline='') as f:
		reader = csv.reader(f)
		result = [list(row) for row in reader]
	return result

def readTSVToList(filename):
	#eprint("Reading data from",filename)
	with open(filename.name, newline='') as f:
		reader = csv.reader(f, delimiter="\t")
		result = [list(row) for row in reader]
	return result


def pearsonr(x, y):
	"""
	Calculates a Pearson correlation coefficient. 
	"""

	assert len(x) == len(y), 'Prediction and gold standard does not have the same length...'

	xm = sum(x)/len(x)
	ym = sum(y)/len(y)

	xn = [k-xm for k in x]
	yn = [k-ym for k in y]

	r = 0 
	r_den_x = 0
	r_den_y = 0
	for xn_val, yn_val in zip(xn, yn):
		r += xn_val*yn_val
		r_den_x += xn_val*xn_val
		r_den_y += yn_val*yn_val

	r_den = sqrt(r_den_x*r_den_y)

	if r_den:
		r = r / r_den
	else:
		r = 0

	# Presumably, if abs(r) > 1, then it is only some small artifact of floating
	# point arithmetic.
	r = max(min(r, 1.0), -1.0)

	return round(r,to_round)

def calculate_pearson(gold, prediction):
	"""
	gold/prediction are a list of lists [ emp pred , distress pred ]
	"""

	# converting to float
	gold_float = []
	for k in gold:
		try:
			gold_float.append(float(k))
		except Exception as e: 
			print(e)
			gold_float.append(0)

	prediction_float = []
	for k in prediction:
		try:
			prediction_float.append(float(k))
		except Exception as e: 
			print(e)
			prediction_float.append(0)

	return pearsonr(gold_float, prediction_float)

def calculate_metrics(golds, predictions, task1, task2, task3, task4):
	"""
	gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
	"""
	
	start_label = 0
	if task1:
		start_label = 0
		gold_empathy = [k[start_label] for k in golds]
		prediction_empathy = [k[start_label] for k in predictions]
		pearson_CONVD = calculate_pearson(gold_empathy, prediction_empathy)
	else:
		pearson_CONVD = 0

	start_label = nb_labels_CONVD
	if task2:
		gold_convt, prediction_convt, pearson_convt = [], [], []
		for i in range(start_label, start_label+nb_labels_CONVT):
			gold_convt.append([k[i] for k in golds])
			prediction_convt.append([k[i] for k in predictions])
			pearson_convt.append(calculate_pearson(gold_convt[-1], prediction_convt[-1]))
			
		avg_pearson_CONVT = sum(pearson_convt)/len(pearson_convt)
		pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy = pearson_convt
	else:
		avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy = 0, 0, 0, 0
	
	start_label += nb_labels_CONVT
	if task3:
		gold_emp, prediction_emp, pearson_emp = [], [], []
		for i in range(start_label, start_label+nb_labels_EMP):
			gold_emp.append([k[i] for k in golds])
			prediction_emp.append([k[i] for k in predictions])
			pearson_emp.append(calculate_pearson(gold_emp[-1], prediction_emp[-1]))
			
		avg_pearson_EMP = sum(pearson_emp)/len(pearson_emp)
		pearson_empathy, pearson_distress = pearson_emp
	else:
		avg_pearson_EMP, pearson_empathy, pearson_distress = 0, 0, 0
	
	start_label += nb_labels_EMP
	if task4:
		gold_per, prediction_per, pearson_per = [], [], []
		for i in range(start_label, start_label+nb_labels_PER):
			gold_per.append([k[i] for k in golds])
			prediction_per.append([k[i] for k in predictions])
			pearson_per.append(calculate_pearson(gold_per[-1], prediction_per[-1]))
			
		avg_pearson_PER = sum(pearson_per)/len(pearson_per)
		person_ope, person_con, person_ext, person_agr, person_sta = pearson_per
	else:
		avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta = 0, 0, 0, 0, 0, 0

	return pearson_CONVD, avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy, avg_pearson_EMP, pearson_empathy, pearson_distress, avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta

def read_file(submission_path, nb_labels=2, nb_samp=10):
	"""
	Read the tsv file
	"""
	# unzipped submission data is always in the 'res' subdirectory
	if not os.path.exists(submission_path):
		print('Could not find submission file {0}'.format(submission_path))
		predictedList = [[0]*nb_labels]*nb_samp
		task = False
	else:
		submission_file = open(os.path.join(submission_path))
		# The 2 first columns
		predictedList = [k[:nb_labels] for k in readTSVToList(submission_file)]
		task = True

	return task, predictedList



def score(input_dir, output_dir):
	# unzipped reference data is always in the 'ref' subdirectory
	# read dev gold standard labels
	truth_file_CONVD = open(os.path.join(input_dir, 'ref', 'goldstandard_CONVD.tsv'))
	goldList_CONVD = [l[:nb_labels_CONVD] for l in readTSVToList(truth_file_CONVD)]
	nb_samp_CONVD = len(goldList_CONVD)

	truth_file_CONVT = open(os.path.join(input_dir, 'ref', 'goldstandard_CONVT.tsv'))
	goldList_CONVT = [l[:nb_labels_CONVT] for l in readTSVToList(truth_file_CONVT)]
	nb_samp_CONVT = len(goldList_CONVT)

	truth_file_EMP = open(os.path.join(input_dir, 'ref', 'goldstandard_EMP.tsv'))
	goldList_EMP = [l[:nb_labels_EMP] for l in readTSVToList(truth_file_EMP)]
	nb_samp_EMP = len(goldList_EMP)

	truth_file_PER = open(os.path.join(input_dir, 'ref', 'goldstandard_PER.tsv'))
	goldList_PER = [l[:nb_labels_PER] for l in readTSVToList(truth_file_PER)]
	nb_samp_PER = len(goldList_PER)

	goldList = [i+j+k+l for i,j,k,l in zip(goldList_CONVD, goldList_CONVT, goldList_EMP, goldList_PER)]

	# read predicyed labels
	submission_path = os.path.join(input_dir, 'res', 'predictions_CONVD.tsv')
	task1, predictedList_CONVD = read_file(submission_path=submission_path, nb_labels=nb_labels_CONVD, nb_samp=nb_samp_CONVD)

	submission_path = os.path.join(input_dir, 'res', 'predictions_CONVT.tsv')
	task2, predictedList_CONVT = read_file(submission_path=submission_path, nb_labels=nb_labels_CONVT, nb_samp=nb_samp_CONVT)

	submission_path = os.path.join(input_dir, 'res', 'predictions_EMP.tsv')
	task3, predictedList_EMP = read_file(submission_path=submission_path, nb_labels=nb_labels_EMP, nb_samp=nb_samp_EMP)

	submission_path = os.path.join(input_dir, 'res', 'predictions_PER.tsv')
	task4, predictedList_PER = read_file(submission_path=submission_path, nb_labels=nb_labels_PER, nb_samp=nb_samp_PER)

	predictedList = [i+j+k+l for i,j,k,l in zip(predictedList_CONVD, predictedList_CONVT, predictedList_EMP, predictedList_PER)]

	if (len(goldList) != len(predictedList)):
		eprint("Number of labels is not aligned!")
		sys.exit(1)

	pearson_CONVD, avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy, avg_pearson_EMP, pearson_empathy, pearson_distress, avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta = calculate_metrics(goldList, predictedList, task1, task2, task3, task4)

	print("Printing results to:", output_dir + '/scores.txt')
	with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
		str_to_write = ''
		# Not sure of that. Useful if the participant want to do only one subtask. Need to see if the leaderboard of the subtask does not update if there are nothing on score.txt 
		if task1:
			str_to_write += "Track 1 (CONVD): Pearson Correlation (perceived empathy): {0}\n".format(pearson_CONVD)
		if task2:
			str_to_write += "Track 2 (CONVT): Averaged Pearson Correlations: {0}\n\tEmotion: {1}\n\tEmotion Polarity: {2}\n\tEmpathy: {3}\n".format(avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy)
		if task3:
			str_to_write += "Track 3 (EMP): Averaged Pearson Correlations: {0}\n\tEmpathy: {1}\n\tDistress: {2}\n".format(avg_pearson_EMP, pearson_empathy, pearson_distress)
		if task4:
			str_to_write += "Track 4 (PER): Averaged Pearson Correlations: {0}\n\tOpenness: {1}\n\tConscientiousness: {2}\n\tExtraversion: {3}\n\tAgreeableness: {4}\n\tStability: {5}\n".format(avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta)
		output_file.write(str_to_write)

def main():
	[_, input_dir, output_dir] = sys.argv
	score(input_dir, output_dir)

if __name__ == '__main__':
	main()