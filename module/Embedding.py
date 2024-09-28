import math
import numpy as np
import utils.Activation as activation
import utils.Preprocessing as processing
import utils.Loss as lossfun

text = open("demofile.txt", "r").read() 
text_token = processing.tokenize(text)
print("tokenize: ", text_token)
idx_to_word, word_to_idx = processing.mapping(text_token)
print("mapping: ", idx_to_word, word_to_idx)
matrix = processing.one_hot_encoding(text_token, word_to_idx, 1)
print("one_hot_encoding len: ", len(matrix))
print("one_hot_encoding: ", matrix)
print("one_hot_encoding[0]: ", matrix[0])
print("one_hot_encoding[1]: ", matrix[1])

class Embedding():
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):
        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden
        print("--------------------------------------------------")
        print("Embedding: ")
        print("--------------------------------------------------")
        print("Input: ", self.size_input)
        print("Hidden: ", self.num_neuron_hidden)
        print("Output: ", self.size_output)

        lower_w1, upper_w1 = -(math.sqrt(6.0) / math.sqrt(size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(size_input + self.size_input))
        self.w1 = np.random.uniform(lower_w1, upper_w1, size=(size_input, self.num_neuron_hidden))
        #self.w1 = np.array(np.random.randn(self.num_neuron_hidden, size_input) * 0.01, dtype=np.float64)
        print("w1: ", self.w1)
        self.w2 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron_hidden, self.size_output))
        #self.w2 = np.array(np.random.randn(self.size_output, self.num_neuron_hidden) * 0.01, dtype=np.float64) 
        print("w2: ", self.w2)
        print("--------------------------------------------------")

    def forward(self, word):
        self.X = word
        self.hidden = word @ self.w1 #np.dot(self.w1, word)

        self.output = self.hidden @ self.w2 #np.dot(self.w2, self.hidden)
  
        return activation.softmax(self.output), self.hidden, self.output

    def backpropogation(self, output, target):
        error_backprop = lossfun.delta_cross_entropy(output, target)

        self.grad_w2 = np.outer(self.hidden.T, error_backprop).T     
        
        #self.grad_w1 = np.dot(error_backprop, self.w2)
        #self.grad_w1 = error_backprop @ self.w2 
        #self.grad_w1 = np.outer(self.X.T, self.grad_w1).T
        #print("shape: ", self.hidden.shape)
        #print("shape: ", error_backprop.shape)

        #self.grad_w2 = self.hidden.T @ error_backprop

        self.grad_w1 = self.X @ self.grad_w2

    def gradient_descend(self):
        self.w1 -= self.learn_rate * self.grad_w1.T
        self.w2 -= self.learn_rate * self.grad_w2.T
    
    def train(self, matrix):
        for ep in range(self.epoch):
            loss = 0
            for elem in matrix:
                softmax_net, hidden_net, output_net = emb.forward(np.array(elem[0]))
                #print("first token: ")
                #print(elem[0])
                #print("context: ")
                
                for y_elem in elem[1]:
                    self.backpropogation(lossfun.cross_entropy(softmax_net, np.array(y_elem)), np.array(y_elem))
                    #print(y_elem)
                    self.gradient_descend()
                    loss += lossfun.cross_entropy(softmax_net, np.array(y_elem))
            print("ep: ", ep)
            print("Loss: ", loss)
            print("machine: ", self.predict(np.array(matrix[word_to_idx["machine"]][0]), matrix, idx_to_word))
            print("implementing: ", self.predict(np.array(matrix[word_to_idx["implementing"]][0]), matrix, idx_to_word))
            print("voice: ", self.predict(np.array(matrix[word_to_idx["voice"]][0]), matrix, idx_to_word)) 
            loss = 0

    def predict(self, word, matrix, indices):
        #word = word.lower()
        out, _, _ = self.forward(word) 

        most_likely_idxs = np.array(out).argsort()[-3:][::-1]
        return [indices[w] for w in most_likely_idxs]

emb = Embedding(0.5, 1000, 145, 40, 145)
emb.train(matrix)


#np.array(matrix[0][1])
#softmax_net, hidden_net, output_net = emb.forward(np.array(matrix[0][0]))
#print(softmax_net)
#print(hidden_net)

#emb.backpropogation(lossfun.cross_entropy(softmax_net, np.array(matrix[0][1])))

#print(lossfun.cross_entropy(softmax_net, np.array(matrix[0][1])))


softmax_net_word1, hidden_net_word1, output_net_word1 = emb.forward(np.array(matrix[word_to_idx["machine"]][0]))
print(hidden_net_word1)

softmax_net_word2, hidden_net_word2, output_net_word2 = emb.forward(np.array(matrix[word_to_idx["learning"]][0]))
print(hidden_net_word2)

cos_sim = np.dot(hidden_net_word1, hidden_net_word2)/(np.linalg.norm(hidden_net_word1)*np.linalg.norm(hidden_net_word2))
print(cos_sim)

softmax_net_word1, hidden_net_word1, output_net_word1 = emb.forward(np.array(matrix[word_to_idx["data"]][0]))
print(hidden_net_word1)

softmax_net_word2, hidden_net_word2, output_net_word2 = emb.forward(np.array(matrix[word_to_idx["voice"]][0]))
print(hidden_net_word2)

cos_sim = np.dot(hidden_net_word1, hidden_net_word2)/(np.linalg.norm(hidden_net_word1)*np.linalg.norm(hidden_net_word2))
print(cos_sim)

