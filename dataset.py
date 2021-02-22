import numpy as np

class XOR:
	def __init__(self,depth):
		self.depth = depth
		self.data = []
		self.label = []
		self.__backtrack_xor([],0)
		self.boundary = (2**depth)//4

	def __backtrack_xor(self,temp,depth):
		if depth == self.depth:
			self.data.append(temp)
			label = 0
			for i in temp:
				label^=i
			cl=[0,1] if label else [1,0]
			self.label.append(cl)
			return 
		for x in [0,1]:
			self.__backtrack_xor(temp+[x],depth+1)
	
	def load_data(self):
		return self.data[:-(self.boundary)],self.label[:-(self.boundary)],self.data[-(self.boundary):],self.label[-(self.boundary):]

class Dataset:
	def __init__(self,depth):
		self.depth = depth

	def load_data(self,gate):
		if gate.lower() == 'xor':
			train_data,train_label,test_data,test_label = XOR(self.depth).load_data()
		return 	train_data,train_label,test_data,test_label	

if __name__ == '__main__':
	obj = Dataset(int(input("Enter number: ")))
	train_data,train_label,test_data,test_label = obj.load_data('xor')		
	print(train_data)
	print(test_data)