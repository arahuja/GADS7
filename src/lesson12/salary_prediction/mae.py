import sys
import csv
import math

if __name__ == '__main__':
	test = csv.DictReader(open(sys.argv[1], 'r'))
	predictions = open(sys.argv[2], 'r')
	t = 0
	i =0
	for row, line in zip(test, predictions):
		t += math.fabs(float(row['SalaryNormalized']) - math.exp(float(line.strip())))
		i += 1
	print t/i	
