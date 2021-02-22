import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt
import sys

class neural_network:

    def __init__(self,n_input,n_neuron,train_data,train_label,test_data,test_label,epochs,alpha):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.output_neuron = 2
        self.n_neuron = n_neuron
        self.weights1 = np.random.uniform(0,1,(self.train_data.shape[1],n_neuron))        
        self.bias1 = np.random.uniform(0,1,(n_neuron))
        self.weights2 = np.random.uniform(0,1,(n_neuron,self.output_neuron))   
        self.bias2 = np.random.uniform(0,1,(self.output_neuron))
        self.epochs = epochs
        self.alpha = alpha 
    
    def sigmoid_activation(self,z):
        return 1/(1+np.exp(-z))

    def hypothesis(self,data,weights,bias):
        return np.dot(data,weights) + bias

    def forward_propagation_1(self,data):        
        self.forward_1 = self.hypothesis(data,self.weights1,self.bias1)
        return self.forward_1   
        
    def forward_propagation_2(self,data):        
        self.forward_1 = self.sigmoid_activation(self.forward_propagation_1(data))     
        self.forward_2 = self.hypothesis(self.forward_1,self.weights2,self.bias2)
        return self.forward_2

    def forward_propagate(self):
        self.final_output = self.sigmoid_activation(self.forward_propagation_2(self.train_data))
        return self.final_output

    def loss_function(self,predicted):      
        return (1/len(predicted))*(-np.sum((self.train_label)*np.log(predicted)+(1-self.train_label)*np.log(1-predicted)))

    def backpropagation(self):
        self.forward_propagate() 
        for data in range(len(self.train_data)):

            Z_final = self.final_output[data]
            Y = self.train_label[data]
            Z1 = self.forward_1[data]
            X = self.train_data[data]
            for i in range(self.weights2.shape[0]):                
                Z1_term = Z1[i]
                for j in range(self.output_neuron):
                    graient = (Z_final[j] - Y[j]) * Z1_term
                    self.weights2[i][j] -= self.alpha*(graient)

                for d in range(self.train_data.shape[1]):
                    gradient_term = 0
                    X_term = X[d]
                    for j in range(self.output_neuron):
                        first_part = (Z_final[j] - Y[j])
                        weights2_term = self.weights2[i][j]
                        third_part = Z1_term*(1-Z1_term)
                        gradient_term += first_part * weights2_term * third_part * X_term
                    self.weights1[d][i] -= self.alpha*(gradient_term)

    def train(self):
        self.cost = []
        for eps in range(self.epochs):
            if eps%10==0:
                predicted = self.forward_propagate()
                loss = self.loss_function(predicted)
                if self.accuracy(eps,test_data,test_label)>0.95:
                    break
                self.cost.append(loss)
            self.backpropagation()                                
            sys.stdout.flush()            
        print('\n')
        
        plt.plot(np.arange(len(self.cost)),self.cost)
        plt.xlabel('No. of epochs')
        plt.ylabel('Cost')
        plt.show()

    def predict(self, data,label):
        vals = self.forward_propagation_2(data)
        predicted = self.sigmoid_activation(vals)
        for i in range(len(predicted)):
            print(np.argmax(predicted[i]),' ==> ',np.argmax(label[i]))
        return predicted             

    def accuracy(self,eps,test_data,target):
        predicted = self.sigmoid_activation(self.forward_propagation_2(test_data))
        count = 0        
        for q in range(len(test_data)):
            if np.argmax(target[q]) == np.argmax(predicted[q]):
                count += 1
        acc=count/len(test_data)
        print("\repoch = {} accuracy = {}".format(eps,acc*100),end=" ")
        return acc                        


if __name__ == "__main__":    

    n_input = int(input("Enter number of input neuron: "))
    n_neuron = int(input("Enter number of neurons in the hidden layer: "))
    lr = float(input('Enter the learning rate:- '))
    epochs = int(input('Enter epochs:- '))

    # Importing Dataset 
    obj = Dataset(n_input)
    train_data,train_label,test_data,test_label = obj.load_data('xor')    

    train_data = np.array(train_data)
    test_data = np.array(test_data)  
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    # Creating the neural network    
    nn = neural_network(n_input,n_neuron,train_data,train_label,test_data,test_label,epochs,lr)
    nn.train()
    nn.predict(test_data,test_label)