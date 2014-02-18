#vw_lda_topics.py
import sys

def loadDictionary(dictFile):
	dictionary = {}
	for line in open(dictFile):
		word, id = line.split('\t')
		dictionary[int(id)] = word.strip()
	return dictionary


if __name__ == '__main__':
	wordMap = loadDictionary(sys.argv[2])
	topicFile = sys.argv[1]
	numWordsPerTopic = int(sys.argv[3])
	skipLines = 11
	skippedLines = 0
	topic_dists = []
	numTopics = 0
	for line in open(topicFile):
		if skippedLines < skipLines:
			skippedLines += 1
			if line.startswith("lda"):
				numTopics = int(line.split(":")[1])
			continue
		topic_dists.append([float(x) for x in line.split()])


	for topic in xrange(numTopics):
		print "="
		print "Topic :", topic
		for score, word_id in sorted(((dist[topic], word) for word, dist in enumerate(topic_dists)), reverse=True)[:numWordsPerTopic]:
			print wordMap[word_id], score



