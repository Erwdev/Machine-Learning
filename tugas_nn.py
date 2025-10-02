import numpy as np 
import math
from tqdm.auto import tqdm
import sys

# ======== Global Constants ========
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class NeuralNetUtils:
    #=======Batching=======
    @staticmethod
    def create_mini_batch(X, y , batch_size, shuffle = True):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            yield X[batch_idx], y[batch_idx]

    @staticmethod
    def create_sgd_batch(X, y, shuffle=True):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            yield X[i:i+1], y[i:i+1]

    @staticmethod
    def create_batch(X, y, shuffle=True):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        yield X[indices], y[indices]

    class Dropout:
        def __init__(self, drop_rate=0.5):
            self.drop_rate = drop_rate
            self.mask = None
    
        def forward(self, X, training=True):
            if training:
                self.mask = (np.random.rand(*X.shape) > self.drop_rate).astype(float)
                return X * self.mask / (1.0 - self.drop_rate)
            else:
                return X

        def backward(self, dA):
            return dA * self.mask / (1.0 - self.drop_rate)
    
    class BatchNorm:
        def __init__(self, input_dim, momentum=0.9, epsilon=1e-8):
            self.gamma = np.ones((1, input_dim))
            self.beta = np.zeros((1, input_dim))
            self.momentum = momentum
            self.epsilon = epsilon
            self.running_mean = np.zeros((1, input_dim))
            self.running_var = np.zeros((1, input_dim))
            self.cache = None

        def forward(self, X, training=True):
            if training:
                batch_mean = np.mean(X, axis=0, keepdims=True)
                batch_var = np.var(X, axis=0, keepdims=True)
                X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
                out = self.gamma * X_norm + self.beta

                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

                self.cache = (X, X_norm, batch_mean, batch_var)
                return out
            else:
                X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
                return self.gamma * X_norm + self.beta

        def backward(self, dout):
            X, X_norm, batch_mean, batch_var = self.cache
            m = X.shape[0]

            dX_norm = dout * self.gamma
            dvar = np.sum(dX_norm * (X - batch_mean) * -0.5 * (batch_var + self.epsilon) ** (-1.5), axis=0)
            dmean = np.sum(dX_norm * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (X - batch_mean), axis=0)

            dX = dX_norm / np.sqrt(batch_var + self.epsilon) + dvar * 2 * (X - batch_mean) / m + dmean / m
            dgamma = np.sum(dout * X_norm, axis=0)
            dbeta = np.sum(dout, axis=0)

            return dX, dgamma, dbeta
            
        
    # ===== Activations =====
    
    @staticmethod
    def linear(x):return x
    @staticmethod
    def linear_derivative(x): return np.ones_like(x)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x):
        s = NeuralNetUtils.sigmoid(x)
        return s * (1-s)
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    @staticmethod
    def leakyRelu(x):
        return np.maximum(0.1*x,x)
    @staticmethod
    def leakyRelu_derivative(x):
        return np.where(x > 0, 1, 0.1)
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
#=======Weight Initialization=======
    @staticmethod
    def random_init(shape):
        return np.random.randn(*shape) * 0.01
    
    @staticmethod
    def he_init(shape):
        fan_in = shape[0]
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * stddev
    
    @staticmethod
    def xavier_init(shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit , limit , size = shape)

class Metrics:
# ===== Losses =====
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2*(y_pred - y_true) / y_true.size
    
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred))
    
    @staticmethod
    def mae_derivative(y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1) / y_true.size
    
    @staticmethod
    def r2_score(y_true, y_pred):

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res/ss_tot)

class NeuralNetwork:
    """A simple feedforward neural network.
    """
    
    def __init__(self):
        self.layers = []
        
        self.loss = None
        self.loss_derivative = None
        self.history = {'loss': [], 'val_loss': [], "metrics":{}}
    
    def add(self, layer):
        self.layers.append(layer)
        
    def set_loss(self, loss_name):
        try:
            self.loss = getattr(Metrics, loss_name)
            self.loss_derivative = getattr(Metrics, f"{loss_name}_derivative")
        except AttributeError:
            raise ValueError(f"Loss function '{loss_name}' not found.")

    def backward(self, y_true, y_pred, learning_rate):
        dA = self.loss_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                dA = layer.backward(dA, learning_rate)
                
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if 'training' in layer.forward.__code__.co_varnames:
                    output = layer.forward(output, training)
                else:
                    output = layer.forward(output)
        return output
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, learning_rate=0.01, verbose=True):
        """Train the neural network with simple, clean output."""
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            
            for X_batch, y_batch in NeuralNetUtils.create_mini_batch(X_train, y_train, batch_size):
                y_pred = self.forward(X_batch, training=True)
                loss = self.loss(y_batch, y_pred)
                epoch_loss += loss
                n_batches += 1
                self.backward(y_batch, y_pred, learning_rate)

            # Calculate epoch metrics
            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)

            # Validation
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = self.loss(y_val, y_val_pred)
                self.history['val_loss'].append(val_loss)

                # Calculate additional metrics
                r2 = Metrics.r2_score(y_val, y_val_pred)
                mae = Metrics.mae(y_val, y_val_pred)
                
                # Store metrics
                if 'metrics' not in self.history:
                    self.history['metrics'] = {}
                self.history['metrics'][epoch] = {"r2": r2, "mae": mae}

                # Print progress every 50 epochs or at the end
                if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                        f"Loss: {epoch_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"R²: {r2:.4f} | "
                        f"MAE: {mae:.6f}")
            else:
                # Training only (no validation)
                if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {epoch_loss:.6f}")
        
        # Final summary
        if verbose:
            print("\n" + "="*80)
            print("Training Complete!")
            if X_val is not None and y_val is not None:
                print(f"Final Training Loss:   {self.history['loss'][-1]:.6f}")
                print(f"Final Validation Loss: {self.history['val_loss'][-1]:.6f}")
                final_r2 = self.history['metrics'][epochs-1]['r2']
                final_mae = self.history['metrics'][epochs-1]['mae']
                print(f"Final R² Score:        {final_r2:.4f}")
                print(f"Final MAE:             {final_mae:.6f}")
            else:
                print(f"Final Training Loss: {self.history['loss'][-1]:.6f}")
            print("="*80)

    def predict(self, X):
        return self.forward(X, training=False)

    def evaluate(self, X_test, y_test, metric='mse'):
            y_pred = self.predict(X_test)
            metric_fn = getattr(Metrics, metric)
            return metric_fn(y_test, y_pred)
    
    def getHistory(self):
        return self.history
    
class Dense:
    """
    A fully connected neural network layer.
    """
    def __init__(self, input_dim, output_dim, activation ='relu', weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # === Initialize Weights & Bias ===
        if weight_init == 'xavier':
            self.W = NeuralNetUtils.xavier_init((input_dim, output_dim))
        elif weight_init == 'he':
            self.W = NeuralNetUtils.he_init((input_dim,output_dim))
        elif weight_init == 'random':
            self.W = NeuralNetUtils.random_init((input_dim, output_dim))
        else:
            raise ValueError("Invalid weight initialization method")

        self.b = np.zeros((1, output_dim))
        
        # === Activation Functions ===
        self.activation = getattr(NeuralNetUtils, activation)
        self.activation_derivative = getattr(NeuralNetUtils, f"{activation}_derivative")
        
        self.input = None
        self.z = None
        
    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        return self.activation(self.z)
    
    def backward(self, dA, learning_rate):
        m = self.input.shape[0]
        #local gradient (derivative) 
        dZ = dA * self.activation_derivative(self.z)
        
        #weight gradient vector update
        dW = np.dot(self.input.T, dZ) / m
        #bias gradient 
        db = np.sum(dZ, axis=0, keepdims= True) / m 
        dA_prev = np.dot(dZ, self.W.T)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dA_prev
