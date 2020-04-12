import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#from dbn import SupervisedDBNClassification
from models import SupervisedDBNRegression

# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target


def split_sequence_univariate(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# define input/training sequence
train_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
valid_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
test_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
 
    
# choose a number of time steps in sliding window
n_steps = 3
# split into samples

X_train_seq, y_train_seq = split_sequence_univariate(train_seq, n_steps)
X_valid_seq, y_valid_seq = split_sequence_univariate(valid_seq, n_steps)
X_test_seq, y_test_seq = split_sequence_univariate(test_seq, n_steps)

X_all = np.append(X_train_seq, X_valid_seq, axis=0) 
X_all = np.append(X_all, X_test_seq, axis=0) 
y_all = np.append(y_train_seq, y_valid_seq, axis=0)
y_all = np.append(y_all, y_test_seq, axis=0) 


## Splitting data
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train_seq)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, y_train_seq)

# Test
X_test = min_max_scaler.transform(X_test_seq)
Y_pred = regressor.predict(X_test)

print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(y_test_seq, Y_pred), mean_squared_error(y_test_seq, Y_pred)))

for i in range(len(y_test_seq)):
    print("Observed: %.3f, Predicted= %.3f" %(y_test_seq[i], Y_pred[i]))