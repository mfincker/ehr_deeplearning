from sys import argv
from collections import Counter
import cPickle as pickle


def csv_to_sequence_data(fStream):
	'''
	Turns Hershel's full_data.csv into a list of tuples, one tuple per visit_id,
	where each tuple has the form: (visit_id, patient_id, [list of codes], label)
	where [list of code idx] is a temporaly sorted list of tuples ((code, code_source), timeToVisitDischarge) 


	Args:
		fStream: Hershel's csv as streamed in by open("file", "r")
		seqPath: path to pickled data file

	Return:
		seqs: list of visits as defined above

	Ex: a visit such as:
		"5, 12, 34, PT, 3, 45, 1
		 5, 15, 30, CT, 3, 45, 1
		 9, 12, 46, CD, 23, 12, 0"

		 is transformed into:
		 [(3,  5, [((34, PT), 33), ((30, CT), 30)], 1),
		  (23, 9, [((46, CD), 0)], 0)]

	Doesn't handle the header row
	'''
	def temporal_sort(seq):
		'''
		Sort a list of tuple based on the descending order of the second element in the tuple
		'''
		return sorted(seq, key=lambda x: -x[1])

	data = {}

	for i, line in enumerate(fStream):
		if len(line) > 0:
			assert len(line.split(',')) == 7, "line " + str(i + 1) + " contains more than 7 columns."
			patient_id, age_in_days, code, code_source, visit_id, age_at_discharge, label = line.strip('\n ').split(',')
			visit = (int(visit_id.strip()), int(patient_id.strip()), int(label.strip()))

			if visit not in data.keys():
				data[visit] = []

			data[visit].append(((int(code.strip()), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))

	seqs = [(visit[0], visit[1], temporal_sort(code_seq), visit[2]) for visit, code_seq in data.items()]

	return seqs

def build_code2idx(fStream, max_code = None, offset = 1):
	'''
	Extract all codes from Hershel's csv and create a dicitonary
	that links (code, code_source) -> index. Additionally, adds an extra code/index pair for "unknown"/0.



	Args:
		fStream: Hershel's csv as streamed in by open("file", "r")
		max_code: number of codes to keep (keeping the most common)

	Return:
	code2idx: {(code, code_source) : index}

	Doesn't deal with header row
	'''
	codes = []
	for line in fStream:
		_, _, code, code_source, _, _, _ = line.strip('\n ').split(',')
		codes.append((int(code), code_source))

	cnt = Counter(codes)
	if max_code:
	    codes = cnt.most_common(max_code)
	else:
	    codes = cnt.most_common()

	code2idx = {code: offset+i for i, (code, _) in enumerate(codes)}

	return code2idx

def preprocess_data(filePath, timeWindow = 180):
	'''
	Turns Hershel's csv into two pickled list saved into files.
	Arg:
		filePath: path to csv file
		timeWindow: keep only codes from within the time window to the visit date
	Return:
		nothing - saves data and labels as pickles object in files
	'''
	def filter_timeWindow(seq, timeWindow):
		''' Returns a subset of the input code list where each code has happened in the given time window from the visit'''
		return [s[0] for s in seq if s[1] <= timeWindow]

	print "Building code index ..."
	code2idx = {}
	with open(filePath, 'r') as f:
		next(f)
		code2idx = build_code2idx(f)
	print "... done!"

	print "Loading visits data ... "
	data = []
	with open(filePath, "r") as f:
		next(f)
		data = csv_to_sequence_data(f)
	"... done!"

	print "Extracting labels"
	labels = [s[-1] for s in data]

	print "Filtering codes based on time window of " + str(timeWindow) + " days"
	data = [filter_timeWindow(s[2], timeWindow) for s in data]
	print "Looking up code indexes"
	data = [[code2idx[code] for code in codes] for codes in data]

	print "Checking data:"
	assert len(data) == len(labels), "Data and lables don't have the same size."
	print "\tnumber of visits: " + str(len(data))
	print "\tmax sequence length: " + str(max([len(d) for d in data]))

	print "Pickling data"
	pickle.dump(data, open("full_data_" + str(timeWindow) + "days_window.pyc", "wb"))
	print "Pickling labels"
	pickle.dump(labels, open("full_labels_" + str(timeWindow) + "days_window.pyc", "wb"))
	print "Pickling code2idx"
	pickle.dump(code2idx, open("code2idx.pyc", "wb"))

	






if __name__ == "__main__":

	# Function Run on toy dataset
	if argv[1] == "csv_to_sequence":
		f = open("dataset/full_data_head.csv")
		next(f) # skip header row
		seqs = csv_to_sequence_data(f)
		f.close()

		for s in seqs:
			print s

	elif argv[1] == "code2idx":
		f = open("dataset/full_data_head.csv")
		next(f) # skip header row
		code2idx = build_code2idx(f)
		f.close()

		for code, index in code2idx.items():
			print str(code) + ": " + str(index)

	elif argv[1] == "preprocess":
		if len(argv) < 3:
			raise IOError, "You must specify a file to process"

		preprocess_data(argv[2], timeWindow = 180)









