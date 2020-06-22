<<<<<<< HEAD
import time
import subprocess, re

def readFile(fileName):
	with open(fileName, 'r') as file:
		return file.read()

def writeFile(fileName, content):
	with open(fileName, 'w') as file:
		file.write(content)

def dataFile(fileName, content):
	with open(fileName, 'a') as file:
		file.write(str(content) + "\n")

def template(content, mapping=[]):
	mapping = {m.name:m.val for m in mapping}
	flag = False
	if content[0] == '$':
		flag = True
	content = content.split('$')
	output = ''
	for c in content:
		if not flag:
			output += c
		elif c.strip() not in mapping:
			raise Exception(f"Could not find {c} in the template mapping")
		else:
			output += str(mapping[c.strip()])
		flag = not flag
	return output

def runRai():
	output = subprocess.run('rai-linux-amd64 -p . --queue rai_amd64_ece408'.split(' '),
		check=True, stderr=subprocess.PIPE).stderr

	output = output.decode("utf-8")
	ops_time = re.findall("Op Time: ([0-9]*\.[0-9]*)", output)
	correctness = re.findall("Correctness: ([0-9]*\.[0-9]*)", output)
	meta_data = re.findall("([0-9]*\.[0-9]*)user ([0-9]*\.[0-9]*)system ([0-9]*:[0-9]*\.[0-9]*)elapsed ([0-9]*%)CPU", output)
	meta_data2 = re.findall("([0-9]*)inputs\+([0-9]*)outputs \(([0-9]*)major\+([0-9]*)minor\)pagefaults ([0-9]*)swaps", output)
	if len(ops_time) != 2:
		print(output)
	return (ops_time, correctness, meta_data, meta_data2)

class Mapping():
	ADD = 'add'
	MULT = 'mult'

	def __init__(self, name, init_val, end_val, incr, method='add'):
		self.name = name
		self.init_val = init_val
		self.end_val = end_val
		if end_val < init_val:
			raise Exception("End value must be larger than initial value")
		self.val = init_val
		self.incr = incr
		self.method = method
		self.nextMapping = None

	def chain(self, other):
		self.nextMapping = other

	def _increment(self):
		val = self.val
		if self.method == Mapping.ADD:
			self.val += self.incr
		elif self.method == Mapping.MULT:
			self.val *= self.incr
		else:
			raise Exception(f"Method {self.method} is not defined")
		return val

	def next(self):
		if self.nextMapping is not None:
			done = self.nextMapping.next()
			if not done:
				return done
			else:
				self.nextMapping.reset()
		self._increment()
		return self.done()

	def values(self):
		values = {self.name:self.val}
		if self.nextMapping is not None:
			values.update(self.nextMapping.values())
		return values

	def done(self):
		return self.val > self.end_val

	def reset(self):
		self.val = self.init_val

def automate(templateName, outputName, delay=120, mapping=[]):
	c = None
	head = None
	for m in mapping:
		if head == None:
			head = m
			c = m
		else:
			c.chain(m)
		c = m
	fileTemplate = readFile(templateName)
	tries = 0
	while not head.done():
		print(f"Sleeping for {delay} seconds")
		time.sleep(delay)
		t = template(fileTemplate, mapping)
		writeFile(outputName, t)
		print("Template completed")
		data = runRai()
		if len(data[0]) != 2:
			print("for" , head.values(), "Failed to run due to len(optimes) =", len(data[0]), " trying again")
			tries += 1
			if tries < 3:
				continue
			else:
				tries = 0
		values = head.values()
		dataFile("./data.txt", (values, data))
		head.next()
	pass


# fileTemplate = readFile('templates/new-forward.cuh.template')
# print(fileTemplate)
# namespace = {'TILE_WIDTH':'8'}

# t = template(fileTemplate, namespace)
mapping = [
	Mapping('TILE_WIDTH', 512, 1024, 2, Mapping.MULT), # Dohun
	# Mapping('TILE_WIDTH', 1, 128, 2, Mapping.MULT), # Michelle
	# Mapping('BLOCK_SIZE', 1, 4096, 2, Mapping.MULT) # Michelle
]

templateFile = 'templates/new-forward.cuh.template'
outFile = 'ece408_src/new-forward.cuh'
automate(templateFile, outFile, mapping=mapping, delay=130)

=======
import time
import subprocess, re

def readFile(fileName):
	with open(fileName, 'r') as file:
		return file.read()

def writeFile(fileName, content):
	with open(fileName, 'w') as file:
		file.write(content)

def dataFile(fileName, content):
	with open(fileName, 'a') as file:
		file.write(str(content) + "\n")

def template(content, mapping=[]):
	mapping = {m.name:m.val for m in mapping}
	flag = False
	if content[0] == '$':
		flag = True
	content = content.split('$')
	output = ''
	for c in content:
		if not flag:
			output += c
		elif c.strip() not in mapping:
			raise Exception(f"Could not find {c} in the template mapping")
		else:
			output += str(mapping[c.strip()])
		flag = not flag
	return output

def runRai():
	output = subprocess.run('../rai -p . --queue rai_amd64_ece408'.split(' '),
		check=True, stderr=subprocess.PIPE).stderr

	output = output.decode("utf-8")
	ops_time = re.findall("Op Time: ([0-9]*\.[0-9]*)", output)
	correctness = re.findall("Correctness: ([0-9]*\.[0-9]*)", output)
	meta_data = re.findall("([0-9]*\.[0-9]*)user ([0-9]*\.[0-9]*)system ([0-9]*:[0-9]*\.[0-9]*)elapsed ([0-9]*%)CPU", output)
	meta_data2 = re.findall("([0-9]*)inputs\+([0-9]*)outputs \(([0-9]*)major\+([0-9]*)minor\)pagefaults ([0-9]*)swaps", output)

	return (ops_time, correctness, meta_data, meta_data2)

class Mapping():
	ADD = 'add'
	MULT = 'mult'

	def __init__(self, name, init_val, end_val, incr, method='add'):
		self.name = name
		self.init_val = init_val
		self.end_val = end_val
		if end_val < init_val:
			raise Exception("End value must be larger than initial value")
		self.val = init_val
		self.incr = incr
		self.method = method
		self.nextMapping = None

	def chain(self, other):
		self.nextMapping = other

	def _increment(self):
		val = self.val
		if self.method == Mapping.ADD:
			self.val += self.incr
		elif self.method == Mapping.MULT:
			self.val *= self.incr
		else:
			raise Exception(f"Method {self.method} is not defined")
		return val

	def next(self):
		if self.nextMapping is not None:
			done = self.nextMapping.next()
			if not done:
				return done
			else:
				self.nextMapping.reset()
		self._increment()
		return self.done()

	def values(self):
		values = {self.name:self.val}
		if self.nextMapping is not None:
			values.update(self.nextMapping.values())
		return values

	def done(self):
		return self.val > self.end_val

	def reset(self):
		self.val = self.init_val

def automate(templateName, outputName, delay=120, mapping=[]):
	c = None
	head = None
	for m in mapping:
		if head == None:
			head = m
			c = m
		else:
			c.chain(m)
		c = m
	fileTemplate = readFile(templateName)
	while not head.done():
		print(f"Sleeping for {delay} seconds")
		time.sleep(delay)
		t = template(fileTemplate, mapping)
		writeFile(outputName, t)
		head.next()
		print("Template completed")
		data = runRai()
		values = head.values()
		dataFile("../data.txt", (values, data))
	pass


# fileTemplate = readFile('templates/new-forward.cuh.template')
# print(fileTemplate)
# namespace = {'TILE_WIDTH':'8'}

# t = template(fileTemplate, namespace)
mapping = [
	Mapping('TILE_WIDTH', 1, 1024, 2, Mapping.MULT),
]
templateFile = 'templates/new-forward.cuh.template'
outFile = 'ece408_src/new-forward.cuh'
automate(templateFile, outFile, mapping=mapping, delay=120)
>>>>>>> SharedMemConv
