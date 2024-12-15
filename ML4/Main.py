import numpy as np

np.set_printoptions(threshold=np.inf)
file1=np.load('test_predictions.npy')
file=np.load('ytrain_Classification2.npy')
print(np.array(file))
#print(np.shape(file1))
num_zeros = np.count_nonzero(file1 == 0)
num_one = np.count_nonzero(file1 == 1)
num_two = np.count_nonzero(file1 == 2)
num_three = np.count_nonzero(file1 == 3)
num_four = np.count_nonzero(file1 == 4)
num_five = np.count_nonzero(file1 == 5)
print(num_zeros,num_one,num_two,num_three,num_four,num_five)
