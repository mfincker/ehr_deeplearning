import sys
import argparse
from collections import Counter
import cPickle as pickle


# def block_to_code_seq(block, visit):
# 	'''
# 	Turns Hershel's full_data.csv into a list of tuples, one tuple per visit_id,
# 	where each tuple has the form: (visit_id, patient_id, [list of codes], label)
# 	where [list of code idx] is a temporaly sorted list of tuples ((code, code_source), timeToVisitDischarge) 

# 	Given Hershel's csv and a visit, go through the csv and returns a tuple corresponding to the visit codes.
# 	Ex: Given the csv:
# 		"5, 12, 34, PT, 3, 45, 1,
# 		 9, 12, 46, CD, 23, 12, 0
# 		 5, 15, 30, CT, 3, 45, 1"

# 		 and the visit (3, 5, 1) (visit_id, patient_id, label)

# 		 returns:
# 		 (visit_id, patient_id, [list of codes: ((code, code_source), timeToDischarge)])
# 		 (3,  5, [((34, PT), 33), ((30, CT), 30)], 1)


# 	Args:
# 		fStream: Hershel's csv as streamed in by open("file", "r")
# 		visit: visit to consider

# 	Return:
# 		seqs: tuple of visit_id, patient_id, list of codes, label

	
# 	Doesn't handle the header row
# 	'''
# 	def temporal_sort(seq):
# 		'''
# 		Sort a list of tuple based on the descending order of the second element in the tuple
# 		'''
# 		return sorted(seq, key=lambda x: -x[1])

# 	codes = []
# 	regex = re.compile(visit[0])

# 	for line in fStream:
# 		if len(line) > 0 and regex.match(line) is not None:
# 			patient_id, age_in_days, code, code_source, visit_id, age_at_discharge, label = line.strip('\n ').split(',')
# 			cur_visit = (visit_id.strip(), patient_id.strip(), int(label.strip()))
# 			if visit == cur_visit:
# 				codes.append(((code.strip(), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))

# 	return (visit[0], visit[1], temporal_sort(codes), visit[2])

# def do_test_csv_to_sequence(args):
# 	csv = """5, 12, 34, PT, 3, 45, 1
# 9, 12, 46, CD, 23, 12, 0
# 5, 10, 30, CT, 3, 45, 1""".split('\n')

# 	output = ("3",  "5", [(("30", "CT"), 35.), (("34", "PT"), 33.)], 1)
# 	output_ = csv_to_sequence_data(csv, ("3", "5", 1))

# 	assert output == output_
# 	print "test csv_to_sequence passed"

def build_code2idx(fStream, max_code = None, offset = 1, returnVisits = False, counter = False):
	'''
	Extract all codes from Hershel's csv and create a dicitonary
	that links (code, code_source) -> index. Additionally, adds an extra code/index pair for "unknown"/0.

	Args:
		fStream: Hershel's csv as streamed in by open("file", "r")
		max_code: number of codes to keep (keeping the most common)
		returnVisits: bool - if True, also return a list of visits [(visit_id, patient_id, label)]
		counter: bool - if True, also return the counter that was used to create the dictionary

	Return:
		code2idx: {(code, code_source) : index}
		counter: counter used to build code2idx
		visits: list of all visits

	Doesn't deal with header row
	'''
	codes = []
	visits = set()
	print "\treading inputFile"
	for line in fStream:
		patient_id, _, code, code_source, visit_id, _, label = line.strip('\n ').split(',')
		codes.append((code, code_source))
		if returnVisits:
			visits.add((visit_id.strip(), patient_id.strip(), int(label.strip())))

	print "\tbuilding counter"
	cnt = Counter(codes)
	if max_code:
		codes = cnt.most_common(max_code)
	else:
		codes = cnt.most_common()

	print "\tbuilding dicitonary"
	code2idx = {code: offset+i for i, (code, _) in enumerate(codes)}
	code2idx[("-1", "UKN")] = 0

	print "\tdone"

	return code2idx, cnt, list(visits)

def do_build_code2idx(args):
	'''Build and save to file the code2idx (visits and or counter) to file'''

	fStream = args.input_file
	next(fStream) # skp header row
	code2idxStream = args.code2idx_file

	counterStream = args.counter_file
	saveCounter = counterStream is not None

	visitsStream = args.visits_file
	saveVisits = visitsStream is not None

	print "Building code2idx / counter / visit list"
	code2idx, counter, visits = build_code2idx(fStream, max_code = args.max_code, offset = 1, returnVisits = saveVisits, counter = saveCounter)

	print "Pickling code2idx ..."
	pickle.dump(code2idx, code2idxStream)
	print "... done"

	if saveCounter:
		print "Pickling counter ..."
		pickle.dump(counter, counterStream)
		print "... done"

	if saveVisits:
		print "Pickling visits ..."
		pickle.dump(visits, visitsStream)
		print "... done"


def preprocess_data(csvStream, visits, code2idx, outDataStream, outLabelStream, timeWindow = 180):
	'''
	Given a code2idx dictionary and a list of visit, transform Hershel's csv into a file where:
		- each line correspond to a visit
		- each line is a sequence of indeces corresponding to the codes, ordered by descending time to timeToDischarge and withtin the timeWindow
		- each index correspond to a code, as described in code2idx dictionary
	'''

	####################
	# Helper functions #
	####################
	def filter_timeWindow(seq, timeWindow):
		''' Returns a subset of the input code list where each code has happened in the given time window from the visit'''
		return [s[0] for s in seq if s[1] <= timeWindow]

	def temporal_sort(seq):
		'''
		Sort a list of tuple based on the descending order of the second element in the tuple
		'''
		return sorted(seq, key=lambda x: -x[1])
	#####################

	n_visit_ = len(visits)
	max_len = 0
	n_visit = 0
	codes = []

	next(csvStream) # skip header row

	# Initialize for first visit
	_, age_in_days, code, code_source, cur_visit, age_at_discharge, label = next(csvStream).strip('\n ').split(',')
	codes.append(((code.strip(), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))

	print "\tNumber of processed visits: " 
	for line in csvStream:
		if len(line) > 0:
			_, age_in_days, code, code_source, visit_id, age_at_discharge, label = line.strip('\n ').split(',')

			# Process cur_visit data and reinitialize for next visit
			if cur_visit != visit_id:
				cur_visit = visit_id

				n_visit = n_visit + 1
				if n_visit % 100 == 0: print "\t" + str(n_visit)

				codes = temporal_sort(codes)

				codes = filter_timeWindow(codes, timeWindow)

				codes = [str(code2idx[c]) for c in codes]

				max_len = max(max_len, len(codes))
				outDataStream.write(" ".join(codes) + "\n")
				outLabelStream.write(str(label) + "\n")

				codes = []

			codes.append(((code.strip(), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))
	# Process last visit
	n_visit = n_visit + 1

	codes = temporal_sort(codes)

	codes = filter_timeWindow(codes, timeWindow)

	codes = [str(code2idx[c]) for c in codes]

	max_len = max(max_len, len(codes))
	outDataStream.write(" ".join(codes) + "\n")
	outLabelStream.write(str(label) + "\n")

	print "done!"
	print "Expected # visits: " + str(n_visit_)
	print "# processed visits: " + str(n_visit)
	print "Max length of code sequence: " + str(max_len)



def do_preprocess_data(args):
	csv = args.csvStream
	print "Loading visits"
	visits = pickle.load(args.visitsStream)
	print "Loading code2idx"
	code2idx = pickle.load(args.code2idxStream)
	out = args.outDataStream
	label = args.outLabelStream
	timeWindow = args.timeWindow

	print "Preprocessing data from: " + str(csv.name) + " wtih parameters:"
	print "- timeWindow: " + str(timeWindow)
	print "- code2idx: " + str(args.code2idxStream.name)
	print "- visits: " + str(args.visitsStream.name)
	print "- data output: " + str(args.outDataStream.name)
	print "- label output: " + str(args.outLabelStream.name)
	print ""
	preprocess_data(csv, visits, code2idx, out, label, timeWindow)


# def preprocess_data(filePath, timeWindow = 180):
# 	'''
# 	Turns Hershel's csv into two pickled list saved into files.
# 	Arg:
# 		filePath: path to csv file
# 		timeWindow: keep only codes from within the time window to the visit date
# 	Return:
# 		nothing - saves data and labels as pickles object in files
# 	'''
# 	def filter_timeWindow(seq, timeWindow):
# 		''' Returns a subset of the input code list where each code has happened in the given time window from the visit'''
# 		return [s[0] for s in seq if s[1] <= timeWindow]

# 	print "Building code index ..."
# 	code2idx = {}
# 	with open(filePath, 'r') as f:
# 		next(f)
# 		code2idx = build_code2idx(f)
# 	print "... done!"
# 	print "Pickling code2idx"
# 	pickle.dump(code2idx, open("code2idx.pyc", "wb"))

# 	print "Loading visits data ... "
# 	data = []
# 	with open(filePath, "r") as f:
# 		next(f)
# 		data = csv_to_sequence_data(f)
# 	"... done!"

# 	print "Extracting labels"
# 	labels = [s[-1] for s in data]

# 	print "Filtering codes based on time window of " + str(timeWindow) + " days"
# 	data = [filter_timeWindow(s[2], timeWindow) for s in data]
# 	print "Looking up code indexes"
# 	data = [[code2idx[code] for code in codes] for codes in data]

# 	print "Checking data:"
# 	assert len(data) == len(labels), "Data and lables don't have the same size."
# 	print "\tnumber of visits: " + str(len(data))
# 	print "\tmax sequence length: " + str(max([len(d) for d in data]))

# 	print "Pickling data"
# 	pickle.dump(data, open("full_data_" + str(timeWindow) + "days_window.pyc", "wb"))
# 	print "Pickling labels"
# 	pickle.dump(labels, open("full_labels_" + str(timeWindow) + "days_window.pyc", "wb"))


	






if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Preprocess Hershel\'s csv data')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('code2idx', help='Build the code2idx dictionary. Additionally, can return list of visits and counter used to build the dictionary.')
	command_parser.add_argument('input_file', type=argparse.FileType('r'), help="input file path - required")
	command_parser.add_argument('code2idx_file', type=argparse.FileType('wb'), help="File path to save pickled code2idx dictionary - required")
	command_parser.add_argument('-c', '--counter_file', type=argparse.FileType('wb'), help="File path to save pickled counter")
	command_parser.add_argument('-v', '--visits_file', type=argparse.FileType('wb'), help="File path to save visits list")
	command_parser.add_argument('-m', '--max_code', type=int, default = None, help="Maximum number of codes to keep")
	command_parser.set_defaults(func=do_build_code2idx)

	# command_parser = subparsers.add_parser("csv_to_seq", help='Returns the code sequence for a given visit.')
	# command_parser.set_defaults(func=do_test_csv_to_sequence)

	command_parser = subparsers.add_parser("preprocess", help="Turn the csv into list of list code indexes and list of labels.")
	command_parser.add_argument('csvStream', type=argparse.FileType('r'), help="Path to the csv containing the data")
	command_parser.add_argument('visitsStream', type=argparse.FileType('rb'), help="Path to the visit list pickle")
	command_parser.add_argument('code2idxStream', type=argparse.FileType('rb'), help="Path to the code2idx dictionary pickle")
	command_parser.add_argument('outDataStream', type=argparse.FileType('w'), help="Path to the output file for the indexes")
	command_parser.add_argument('outLabelStream', type=argparse.FileType('w'), help='Path to the output file for the labels')
	command_parser.add_argument('-t', '--timeWindow', type=float, default = 180, help="Time window in days to include codes")
	command_parser.set_defaults(func=do_preprocess_data)

	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)








