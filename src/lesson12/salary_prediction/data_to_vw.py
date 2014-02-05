import sys
import math
import csv


if __name__ == '__main__':
	f = csv.DictReader(open(sys.argv[1], 'r'))
	for row in f:
		for (k, v) in row.items():
			row[k] = v.replace(":", "")
		print str(math.log(float(row['SalaryNormalized']))) \
			+ " |location " + "+".join(row['LocationNormalized'].split()) \
			+ " |l " + " ".join( x.lower() for x in row['LocationRaw'].split()) \
			+ " |c " + "+".join(row['Category'].split()) \
			+ " |ctype " + row['ContractType'] \
			+ " |ctime " + row['ContractTime'] \
			+ " |company " + "+".join(row['Company'].split()) \
			+ " |title2 " + "+".join(row['Title'].split()) \
			+ " |t " + row['Title'] \
	#		+ " |d " + " ".join(x.lower() for x in row['FullDescription'].split())
