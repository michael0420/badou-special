import numpy as np
import scipy
from keras.datasets import mnist
from keras.utils import to_categorical
from tqdm import tqdm

class NeuralNetWork():
    def __init__(self, num_input, num_hidden, num_output, lr) -> None:
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        
        self.lr = lr
        
        self.wIh = np.random.rand(num_input, num_hidden) - 0.5
        self.wHo = np.random.rand(num_hidden, num_output) - 0.5
        
        self.activation = lambda x:scipy.special.expit(x)
    
    
    def fit(self, train_features, train_labels, epochs, batch_size):
        def _get_iter(features, labels, batch_size):
            indices = list(range(len(features)))
            np.random.shuffle(indices)
            for i in range(0, len(features), batch_size):
                yield (features[i:i+batch_size], labels[i:i+batch_size])
        
        for epoch in range(epochs):
            train_loss = []
            train_iter = _get_iter(train_features, train_labels, batch_size)
            for X, y in tqdm(train_iter):
                y_hat, o_h = self.forward(X)
                loss = (y_hat - y) ** 2 / 2
                train_loss.append(loss.mean())
                
                error = (y_hat - y) / (self.num_output *  batch_size) * 2 * y_hat*(1-y_hat)
                hidden_error = error @ self.wHo.T
                g_c1 = o_h.T @ error
                g_c2 = X.T @ (hidden_error * o_h * (1-o_h))
                
                self.wHo -= self.lr * g_c1
                self.wIh -= self.lr * g_c2
            print(f'epoch: {epoch+1}, train loss: {sum(train_loss) / len(train_loss)}')
    
    def forward(self, x):
        x = x @ self.wIh
        o_h = self.activation(x)
        y = o_h @ self.wHo
        y = self.activation(y)
        return y, o_h
    
    def predict(self, x):
        y,_ = self.forward(x)
        return y
    

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
train_features = train_features.reshape(len(train_features), -1)
train_features = train_features.astype(np.float32) / 255
test_features = test_features.reshape(len(test_features), -1)
test_features = test_features.astype(np.float32) / 255

num_input, num_hidden, num_output, lr = train_features.shape[1], 512, 10, 0.3
epochs, batch_size = 10, 128

net = NeuralNetWork(num_input, num_hidden, num_output, lr)
net.fit(train_features, train_labels, epochs, batch_size)
y_hat = net.predict(test_features)
y_hat = np.argmax(y_hat, axis=1)
acc = (y_hat == test_labels).sum() / y_hat.shape[0]
print(f'acc on test set is: {acc}')