#!/usr/bin/env python

# Specific to adzuna salary data processing.
# Usage:
# (input from stdin, output to stdout)
# (reads complete test data from hard-coded location)
# Reports MAE for class test set.

import csv
import fileinput

# Read in truth:
readfile = file("../../test-full.csv", "r")
reader = csv.reader(readfile)
reader.next() # skip header
truth = {}
for line in reader:
  truth[line[0]] = int(line[10])
readfile.close()

# read in answers
reader = csv.reader(fileinput.input())
reader.next()
answer = {}
for line in reader:
  answer[line[0]] = float(line[1])

if len(answer) != 5000: print "NOT 5,000 PREDICTIONS!"

sum_error = 0
for key, value in answer.iteritems():
  sum_error += abs(value - truth[key])

print str(sum_error / 5000)
