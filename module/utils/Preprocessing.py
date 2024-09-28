import re

def tokenize(text):
  pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
  return pattern.findall(text.lower())

def mapping(text_array):
  idx_to_word = {}
  word_to_idx = {}

  i=0; j=0
  for idx, word in enumerate(text_array):
    if word not in idx_to_word.values():
      idx_to_word[i] = word
      i+=1
      
    
    if word not in word_to_idx.keys():
      word_to_idx[word] = j
      j+=1
      

  return idx_to_word, word_to_idx


def one_hot_encoding(text_array, word_to_idx, window_size):

  matrix = []

  for idx, word in enumerate(text_array):
    
    center_vec = [0 for w in word_to_idx]
    center_vec[word_to_idx[word]] = 1


    context_vec = []
    for i in range(-window_size, window_size+1):
      
      if i == 0 or idx+i < 0 or idx+i >= len(text_array) or word == text_array[idx+i]:
        continue
      
      temp = [0 for w in word_to_idx]
      temp[word_to_idx[text_array[idx+i]]] = 1 

      context_vec.append(temp)
      
    matrix.append([center_vec, context_vec])
    

  return matrix

"""
def one_hot_encoding(text_array, word_to_idx, window_size):
  matrix = []

  for idx, word in enumerate(text_array):
    
    center_vec = [0 for w in word_to_idx]
    center_vec[word_to_idx[word]] = 1

    context_vec = []
    
    if idx == 0:
        for i in range(1, window_size+1):
            temp = word_to_idx[text_array[idx+i]] = 1 
            context_vec.append(temp)
    
    if idx == len(text_array):
        for i in range(1, window_size+1):
            temp = word_to_idx[text_array[idx-i]] = 1 
            context_vec.append(temp)
    else:
        for i in range(-window_size, window_size+1):
            if i == 0 or idx+i < 0 or idx+i >= len(text_array) or word == text_array[idx+i]:
                continue

            temp = [0 for w in word_to_idx]
            temp[word_to_idx[text_array[idx+i]]] = 1 

            context_vec.append(temp)
"""
