# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
import numpy as np

def load_dataset(filename):
    dataset = []
    with open(filename ,'r') as input_file:
        csv_reader = reader(input_file)
        for row in csv_reader:
            dataset.append(row)
        return dataset

def dataset_cleanse(dataset):
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

def train_coefficients(training_data,n_epochs,learning_rate):
    coefficients = [0 for i in range(len(training_data[0]))]
    for epoch in range(n_epochs):
        for i,coefficient in enumerate(coefficients):
            for j,row in enumerate(training_data):
                theta = np.matrix(
                    [coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
                     coefficients[5],
                     coefficients[6],
                     coefficients[7], coefficients[8]])

                x = np.matrix(
                    [1.0, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                     float(row[6]), float(row[7])])
                theta_times_x = float(np.matmul(theta, x.transpose()))
                h = 1.0 / (1.0 + exp(-1 * theta_times_x))
                y = float(row[8])
                #Coefficient 0 has a different learning algorithm to other coefficients
                if i == 0:
                    coefficients[i] = coefficients[i] + learning_rate * (-1 * y - h) * h * (1.0 - h)
                    continue
                else:
                    xj = row[i-1]
                    coefficients[i] = coefficients[i] - (learning_rate * (h - y) * xj)
    return coefficients

#minmax normalisation
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

#Convert String data point to floats
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

#Determining Accuracy of training model - returns amount of correct points
def predict(coefficients,dataset):
    theta = np.array(
        [coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5],
         coefficients[6], coefficients[7],coefficients[8]])
    correct = 0
    for row in dataset:
        x = np.array([1.0,float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])])
        h = 1.0 / (1.0 + exp(-1 * float(np.matmul(theta, x))))
        y = float(row[8])
        # print 'predicted {}  -  actual {}'.format(round(h),y)
        if y == round(h):
            correct += 1

    accuracy = float(correct) / float(len(dataset))
    return accuracy

def cross_validation_split(dataset,k_folds):

    dataset_copy = list(dataset)
    folds = []
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = []
        while len(fold) < fold_size:
            fold.append(dataset_copy.pop(randrange(len(dataset_copy))))
        folds.append(fold)
        if i ==0:
            assert(len(fold) == int(len(dataset) / k_folds)),'Incorrect Fold Size'
    return folds

def cross_validation(dataset,k_folds,n_epochs,learning_rate):
    folds = cross_validation_split(dataset,k_folds)
    scores = []

    for i,fold in enumerate(folds):
        training_dataset = []
        run = 'K-Fold ' + str(i)
        if i ==0:
            for train_fold in folds[i+1:]:
                training_dataset.extend(train_fold)
            test_dataset = folds[i]

        elif i == len(folds):
            for train_fold in folds[:i-1]:
                training_dataset.extend(train_fold)
            test_dataset = folds[i]
        else:
            for train_fold in folds[:i-1]:
                training_dataset.extend(train_fold)
            for train_fold in folds[i+1:]:
                training_dataset.extend(train_fold)
            test_dataset = folds[i]
        coefficients = train_coefficients(training_dataset, n_epochs, learning_rate)
        accuracy = predict(coefficients, test_dataset)
        score = (accuracy, run)
        scores.append(score)

    acc_total = 0
    for accuracy,run in scores:
        acc_total += accuracy
        print accuracy,run
    print 'average accuracy is {}'.format(acc_total/float(k_folds))

def main():
    dataset = load_dataset(r'C:\xxx\Documents\Machine Learning\ML Data\pima_diabetes.txt')
    dataset_cleanse(dataset)
    normalize_dataset(dataset,dataset_minmax(dataset))
    cross_validation(dataset,10,100,0.5)

main()

