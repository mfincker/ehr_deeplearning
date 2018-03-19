import sys
import argparse
from collections import Counter
import cPickle as pickle
import logging
from sklearn.model_selection import train_test_split

consoleHandler = logging.StreamHandler(sys.stdout)
fileHandler = logging.FileHandler("preprocess.log")

logger = logging.getLogger()
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

logger.setLevel(logging.DEBUG)


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
	logger.info("\treading inputFile")
	for line in fStream:
		patient_id, _, code, code_source, visit_id, _, label = line.strip('\n ').split(',')
		codes.append((code, code_source))
		if returnVisits:
			visits.add((visit_id.strip(), patient_id.strip(), int(label.strip())))

	logger.info("\tbuilding counter")
	cnt = Counter(codes)
	if max_code:
		codes = cnt.most_common(max_code)
	else:
		codes = cnt.most_common()

	logger.info("\tbuilding dicitonary")
	code2idx = {code: offset+i for i, (code, _) in enumerate(codes)}
	code2idx[("-1", "UKN")] = 0

	logger.info("\tdone")

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

	logger.info("Building code2idx / counter / visit list")
	code2idx, counter, visits = build_code2idx(fStream, max_code = args.max_code, offset = 1, returnVisits = saveVisits, counter = saveCounter)

	logger.info("Pickling code2idx ...")
	pickle.dump(code2idx, code2idxStream)
	logger.info("... done")

	if saveCounter:
		logger.info("Pickling counter ...")
		pickle.dump(counter, counterStream)
		logger.info("... done")

	if saveVisits:
		logger.info("Pickling visits ...")
		pickle.dump(visits, visitsStream)
		logger.info("... done")

def build_code2idx_from_counter(counter, max_code = None, offset = 1):
	'''Build code2idx from a preloaded counter'''
	if max_code:
		codes = counter.most_common(max_code)
	else:
		codes = counter.most_common()

	logger.info("\tbuilding dicitonary with " + str(max_code) + " codes.")
	code2idx = {code: offset+i for i, (code, _) in enumerate(codes)}
	code2idx[("-1", "UKN")] = 0

	logger.info("\tdone")

	return code2idx

def do_build_code2idx_from_counter(args):
	counter = pickle.load(args.counter_file)
	max_code = args.max_code
	code2idxStream = args.code2idx_file

	"Build index dictionary from preloaded counter"
	code2idx = build_code2idx_from_counter(counter, max_code, offset = 1)

	logger.info("Pickling code2idx ...")
	pickle.dump(code2idx, code2idxStream)
	logger.info("... done")


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
	all_codes = []
	all_labels = []
	codes = []
	label_ = -1

	next(csvStream) # skip header row

	# Initialize for first visit
	_, age_in_days, code, code_source, cur_visit, age_at_discharge, label = next(csvStream).strip('\n ').split(',')
	codes.append(((code.strip(), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))
	label_ = label

	logger.info("\tNumber of processed visits: " )
	for line in csvStream:
		if len(line) > 0:
			_, age_in_days, code, code_source, visit_id, age_at_discharge, label = line.strip('\n ').split(',')

			# Process cur_visit data and reinitialize for next visit
			if cur_visit != visit_id:
				cur_visit = visit_id

				n_visit = n_visit + 1
				if n_visit % 1000 == 0: logger.info("\t" + str(n_visit))

				codes = temporal_sort(codes)

				codes = filter_timeWindow(codes, timeWindow)

				codes = [code2idx.get(c, code2idx[("-1", "UKN")]) for c in codes]
				all_codes.append(codes)
				all_labels.append(int(label_))
				max_len = max(max_len, len(codes))

				codes = []
				labels = -1

			codes.append(((code.strip(), code_source.strip()), float(age_at_discharge.strip()) - float(age_in_days.strip())))
			label_ = label
	# Process last visit
	n_visit = n_visit + 1

	codes = temporal_sort(codes)

	codes = filter_timeWindow(codes, timeWindow)

	codes = [code2idx.get(c, code2idx[("-1", "UKN")]) for c in codes]
	all_codes.append(codes)
	all_labels.append(int(label_))
	max_len = max(max_len, len(codes))

	assert len(all_codes) == len(all_labels), "# labels (" + str(len(all_labels)) + ") != # code sequences (" + str(len(all_codes)) + ")"
	# Save as pickle
	pickle.dump(all_codes, outDataStream)
	pickle.dump(all_labels, outLabelStream)

	logger.info("done!")
	logger.info("Expected # visits: " + str(n_visit_))
	logger.info("# processed visits: " + str(n_visit))
	logger.info("Max length of code sequence: " + str(max_len))



def do_preprocess_data(args):
	csv = args.csvStream
	logger.info("Loading visits")
	visits = pickle.load(args.visitsStream)
	logger.info("Loading code2idx")
	code2idx = pickle.load(args.code2idxStream)
	out = args.outDataStream
	label = args.outLabelStream
	timeWindow = args.timeWindow

	logger.info("Preprocessing data from: " + str(csv.name) + " wtih parameters:")
	logger.info("- timeWindow: " + str(timeWindow))
	logger.info("- code2idx: " + str(args.code2idxStream.name))
	logger.info("- visits: " + str(args.visitsStream.name))
	logger.info("- data output: " + str(args.outDataStream.name))
	logger.info("- label output: " + str(args.outLabelStream.name))
	logger.info("")
	preprocess_data(csv, visits, code2idx, out, label, timeWindow)


def split_train_dev_test(full_data, full_label, train, dev, test):
	''' Split the full_data (list of list) and full_label 
	(of same length as full_data) into train / dev / test sets
	of corresponding to train / dev / test percentage of the full_data.

	Args:
		- full_data: list of list of codes, one item per code sequence
		- full_label: list of labels (same lenght as full_data)
		- train / dev / set: proportion split of the full_data into train / dev / test

	Return:
		train_x, train_y, dev_x, dev_y, test_x, test_y
	'''
	assert abs(train + dev + test  - 1) <= 0.001, "Train / dev / test proportions don't sum to 1."
	eval_prop = test + dev
	train_x, eval_x, train_y, eval_y = train_test_split(full_data, full_label, test_size = eval_prop)
	dev_x, test_x, dev_y, test_y = train_test_split(eval_x, eval_y, test_size = test / eval_prop)

	return train_x, train_y, dev_x, dev_y, test_x, test_y


def do_split_train_dev_test(args):
	'''Read in the full_data, split it into train/dev/test sets and 
	save them as pickles.
	'''
	full_data = []
	full_labels = []

	print "Loading data"
	full_data = pickle.load(args.dataFile)

	print "Loading labels"
	full_labels = pickle.load(args.labelFile)

	train = args.train
	dev = args.dev
	test = args.test

	print "Splitting"
	train_x, train_y, dev_x, dev_y, test_x, test_y = split_train_dev_test(full_data, full_labels, train, dev, test)

	assert len(train_y) + len(test_y) + len(dev_y) == len(full_labels), "Split issue: sum of train/test/dev length doesn't match length of full data."

	root = args.dataFile.name.split('.')[:-1]
	root = "".join(root)

	print "Pickling train"
	with open(root + ".train_x.pyc", 'wb') as f:
		pickle.dump(train_x, f)
	with open(root + ".train_y.pyc", 'wb') as f:
		pickle.dump(train_y, f)

	print "Pickling dev"
	with open(root + ".dev_x.pyc", 'wb') as f:
		pickle.dump(dev_x, f)
	with open(root + ".dev_y.pyc", 'wb') as f:
		pickle.dump(dev_y, f)

	print "Pickling test"
	with open(root + ".test_x.pyc", 'wb') as f:
		pickle.dump(test_x, f)
	with open(root + ".test_y.pyc", 'wb') as f:
		pickle.dump(test_y, f)

# def aggregate(data, labels, counter, code2idx, num_idx = 1000, too_common_idx = 40):
# 	'''Given a list of list of code indices in the rnn feed format, 
# 	returns a list of list of len num_idx with the count of each codes 
# 	encountered in this visit''' 

# 	idx_to_remove = [code2idx[c] for c, _ in counter.most_common(too_common_idx)]
# 	idx_to_remove.append(0)
# 	print idx_to_remove

# 	idx_to_keep = [code2idx[c] for c, _ in counter.most_common(num_idx) if code2idx[c] not in idx_to_remove]
# 	new_idx_map = {idx: i for i, idx in enumerate(idx_to_keep)} 


# 	data = [[new_idx_map[c] for c in seq if c in idx_to_keep] for seq in data]

# 	zero_length = [i for i, seq in enumerate(data) if len(seq) == 0]
# 	data = [seq for i, seq in enumerate(data) if i not in zero_length]
# 	labels = [l for i, l in enumerate(labels) if i not in zero_length]

# 	aggregated_data = []
# 	for seq in data:
# 		aggregate_visit = [0] * num_idx
# 		for c in seq:
# 			aggregate_visit[c] = aggregate_visit[c] + 1
# 		aggregated_data.append(aggregate_visit)

# 	return aggregated_data, labels

def aggregate(data, labels, num_idx = 1000, too_common_idx = 40):
	'''Given a list of list of code indices in the rnn feed format, 
	returns a list of list of len num_idx with the count of each codes 
	encountered in this visit''' 
	print data
	print labels

	idx_to_remove = too_common_idx
	idx_to_keep = num_idx + too_common_idx

	data = [[c for c in seq if c > idx_to_remove and c <= idx_to_keep] for seq in data]

	zero_length = [i for i, seq in enumerate(data) if len(seq) == 0]
	data = [seq for i, seq in enumerate(data) if i not in zero_length]
	labels = [l for i, l in enumerate(labels) if i not in zero_length]

	print data
	print labels

	aggregated_data = []
	for seq in data:
		aggregate_visit = [0] * num_idx 
		for c in seq:
			aggregate_visit[c - idx_to_remove - 1] = aggregate_visit[c - idx_to_remove - 1] + 1
		aggregated_data.append(aggregate_visit)

	return aggregated_data, labels

def do_aggregate_test(args):
	x = [[0, 10, 12, 1, 500, 250],
		[0, 0],
		[1040, 1041]]
	y = [1, 1]

	# counter = pickle.load(args.counter)
	# code2idx = pickle.load(args.code2idx)
	num_idx = args.num_idx
	too_common_idx = args.too_common_idx

	data, labels = aggregate(x, y, num_idx, too_common_idx)

	print data[0][460 - 1]
	print data[0][210 - 1]
	print data[1][1000 - 1]
	print data[2]



def do_aggregate(args):
	x = pickle.load(args.x)
	y = pickle.load(args.y)
	# counter = pickle.load(args.counter)
	# code2idx = pickle.load(args.code2idx)
	num_idx = args.num_idx
	too_common_idx = args.too_common_idx

	data, labels = aggregate(x, y, counter, code2idx)

	with open(".".join(x.name.slpit('.')[:-1]) + ".aggregated.tsv") as f:
		for seq in data:
			seq = [str(s) for s in seq]
			f.write("\t".join(seq) + "\n")

	with open(".".join(y.name.slpit('.')[:-1]) + ".aggregated.tsv") as f:
		for l in labels:
			f.write(str(l) + "\n")
		





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

	command_parser = subparsers.add_parser('code2idx_from_counter', help='Build the code2idx dictionary from a preloaded counter.')
	command_parser.add_argument('counter_file', type=argparse.FileType('rb'), help="counter file path - required")
	command_parser.add_argument('code2idx_file', type=argparse.FileType('wb'), help="File path to save pickled code2idx dictionary - required")
	command_parser.add_argument('-m', '--max_code', type=int, default = None, help="Maximum number of codes to keep")
	command_parser.set_defaults(func=do_build_code2idx_from_counter)	

	# command_parser = subparsers.add_parser("csv_to_seq", help='Returns the code sequence for a given visit.')
	# command_parser.set_defaults(func=do_test_csv_to_sequence)

	command_parser = subparsers.add_parser("preprocess", help="Turn the csv into list of list code indexes and list of labels.")
	command_parser.add_argument('csvStream', type=argparse.FileType('r'), help="Path to the csv containing the data")
	command_parser.add_argument('visitsStream', type=argparse.FileType('rb'), help="Path to the visit list pickle")
	command_parser.add_argument('code2idxStream', type=argparse.FileType('rb'), help="Path to the code2idx dictionary pickle")
	command_parser.add_argument('outDataStream', type=argparse.FileType('wb'), help="Path to the output file for the indexes")
	command_parser.add_argument('outLabelStream', type=argparse.FileType('wb'), help='Path to the output file for the labels')
	command_parser.add_argument('-t', '--timeWindow', type=float, default = 180, help="Time window in days to include codes")
	command_parser.set_defaults(func=do_preprocess_data)

	command_parser = subparsers.add_parser("split", help="Split the formatted indexed data into split / train / test sets and pickle them.")
	command_parser.add_argument('dataFile', type=argparse.FileType('rb'), help="Path to the data file")
	command_parser.add_argument('labelFile', type=argparse.FileType('rb'), help="Path to the label file")
	command_parser.add_argument('-t', '--train', type = float, default = 0.8, help="Proportion of data in train set - default 0.8")
	command_parser.add_argument('-d', '--dev', type = float, default = 0.1, help="Proportion of data in dev set - default 0.1")
	command_parser.add_argument('-s', '--test', type = float, default = 0.1, help="Proportion of data in test set - default 0.1")
	command_parser.set_defaults(func=do_split_train_dev_test)

	command_parser = subparsers.add_parser("aggregate")
	command_parser.add_argument("rnn_data", type = argparse.FileType('rb'), help = "Path to the rnn data to aggregate")
	command_parser.add_argument("rnn_label", type = argparse.FileType('rb'), help ="Path to the rnn labels to aggregate")
	# command_parser.add_argument("counter", type = argparse.FileType('rb'), help = "path to the counter")
	# command_parser.add_argument("code2idx", type = argparse.FileType('rb'), help = "path to the code2idx dict")
	command_parser.add_argument("-n", "--num_idx", type = int, default = 1000, help = "Number of codes to keep")
	command_parser.add_argument("--too_common_idx", type = int, default = 40, help = "Number of most common indices to remove")
	command_parser.set_defaults(func=do_aggregate)

	command_parser = subparsers.add_parser("test_aggregate")
	# command_parser.add_argument("rnn_data", type = argparse.FileType('rb'), help = "Path to the rnn data to aggregate")
	# command_parser.add_argument("rnn_label", type = argparse.FileType('rb'), help ="Path to the rnn labels to aggregate")
	# command_parser.add_argument("counter", type = argparse.FileType('rb'), help = "path to the counter")
	# command_parser.add_argument("code2idx", type = argparse.FileType('rb'), help = "path to the code2idx dict")
	command_parser.add_argument("-n", "--num_idx", type = int, default = 1000, help = "Number of codes to keep")
	command_parser.add_argument("--too_common_idx", type = int, default = 40, help = "Number of most common indices to remove")
	command_parser.set_defaults(func=do_aggregate_test)

	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)








