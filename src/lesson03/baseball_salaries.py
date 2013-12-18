#!/usr/bin/env python
# Read in the file
# For each line
  # Extract player ID
  # Extract the salary
  # Save each salary for player

#For each player
  # Compute an average


f = open('/Users/arahuja/Downloads/lahman-csv_2013-12-10/Salaries.csv')
salaries = {}
f.readline()
for line in f:
  fields = line.strip().split(",")
  playerID = fields[-2]
  salary = int(fields[-1])
##NOT FINE
##  salaries[playerID] = salaries.get(playerID, []).append(salary)
##
#  salaries.get(playerID, []).append(salary)
  salaries[playerID] = salaries.get(playerID, []) + [salary]
#print salaries.items()
for playerID, salaryList in salaries.items():

  print playerID, float(sum(salaryList))/len(salaryList)
#for playerID in salaries.keys():
#  print playerID, float(sum(salaries[playerID]))/len(salaries[playerID])
