
def parse_line(line_list):
	for c_i in range(len(line_list)):
		c = line_list[c_i]
		if c != '0' and c != '1':
			return []
		line_list[c_i] = int(line_list[c_i])	
	return line_list

def import_data(filename):
	lines = []
	with open(filename, "r") as f:
		for line in f.readlines():
			# Only include lines related to the activemask
			if(len(line) != 33):
				continue

			result = parse_line(list(line)[:-1])
			#print(result)
			if result != []:
				lines.append(result)
	return lines

def TBC(lines):
	initial_counter = 0
	for i in range(len(lines)-1):
		if not any(lines[i]):
			initial_counter += 1
	#TBC
	for i in range(len(lines)-1):
	#for i in range(100):
		if all(lines[i]): 
			i += process_line(i, lines)

	counter = 0
	#for i in range(100):
	pruned_output = []
	for i in range(len(lines)-1):
		if not any(lines[i]):
			counter += 1
		else:
			pruned_output.append(lines[i])
	simd_counter = 0
	for line in pruned_output:
		for c in line:
			#print(c)
			if c == 1:
				simd_counter += 1
	print(simd_counter, len(pruned_output), len(pruned_output[0]))
	SIMD_util = simd_counter/(len(pruned_output)*len(pruned_output[0]))
	print(SIMD_util)
	print(len(pruned_output), len(lines)-1, initial_counter, counter)


def find_range(i, lines):
	branch_point = i
	reconvergence_point = i
	for k in range(i+1, len(lines)-1):
		if all(lines[k]):
			reconvergence_point = k
			break
	if reconvergence_point - branch_point != 0 and reconvergence_point - branch_point != 1: 
		#print(branch_point, reconvergence_point)
		#print(lines[branch_point], lines[reconvergence_point])
		return branch_point, reconvergence_point
	return branch_point, branch_point

def compact_lane(c_i, lines, process_line, branch_p, recon_p):
	for l in range(1, recon_p - branch_p):
		next_line = lines[l+branch_p]

		# Compact lane if compaction is available
		if next_line[c_i] == 1 and process_line[c_i] == 0:
			# Swap
			process_line[c_i], next_line[c_i] = next_line[c_i], process_line[c_i]
			return
	

def process_line(i, lines):
	branch_p, recon_p = find_range(i, lines)

	line_length = 32
	# Try to compact every line
	for p in range(branch_p, recon_p):
		process_line = lines[p]

		# For every lane in processing line
		for c_i in range(line_length):

			# No need to compact lane
			if process_line[c_i] == 1:
				continue
			compact_lane(c_i, lines, process_line, branch_p, recon_p)

	return recon_p - branch_p


def main():
	#filename = "active_input.txt"
	filename = "BFSoutput.txt"
	#filename = "active_small.txt"
	lines = import_data(filename)
	#print(lines)
	TBC(lines)

main()

