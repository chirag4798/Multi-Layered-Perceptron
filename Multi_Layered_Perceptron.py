
from utility_functions import *

class Multi_Layered_Perceptron:
    
    '''
    A Multi Layered Perceptron model.

    Attributes:
        n_inputs            : Number of neurons in the input layer (int; default=8).
        hidden_layers       : Number of neurons in each hidden layers (list of int; default=[8, 6, 4]).
        n_outputs           : Number of neurons in the output layer (int; default=2).
        activation_function : Activation function to use, posiible values 'sigmoid' and 'relu'. (Note: relu will be added in the future hopefully)
        random_state        : Value given to np.random.seed() (int; default=1).
    Returns:
        Initializes a Multi Layered Perceptron model, 
        with sigmoid activation function and Cross Etropy Loss as its objective function.
    Methods:
        1. GradientDescent(self, X, y)
        2. BatchSGD(self, X, y)
        3. predict_proba(self, X)
        4. preict(self, X)
        5. plot_loss(self)
    '''
    
    def __init__(self, n_inputs=8, hidden_layers=[8, 6, 4], n_outputs=2, activation_function='sigmoid', random_state=1): # Default values suitable for our problem

        '''
        Constructer for the Multi_Layered_Perceptron object.
        '''
        
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.activation_function = activation_function
        self.random_state = random_state
        Layers = [n_inputs] + hidden_layers + [n_outputs]
        self.losses = []
        self.initialize_weights_derivatives(Layers)
        self.initialize_activations(Layers)
        
    def initialize_weights_derivatives(self, layers):

        '''
        Initialize Weights and Biases Tensors.
        
        Args:
            layers: List of int describing the each layer.
        '''
        
        np.random.seed(0)
        W, dW, B, dB, M, V = [],[],[],[],[],[]    
        for i in range(1,len(layers)):
            np.random.seed(self.random_state)
            w, dw = np.random.normal(loc=0, scale=0.01,	size=(layers[i-1], layers[i])), np.zeros((layers[i-1], layers[i]))
            b, db = np.random.rand(layers[i]), np.zeros(layers[i])
            m, v = np.zeros_like(w), np.zeros_like(w)
            W.append(w);dW.append(dw);B.append(b);dB.append(db);M.append(m);V.append(v)
        self.W, self.dW, self.B, self.dB, self.M, self.V =  W, dW, B, dB, M, V
        
    def initialize_activations(self, layers):
        
        '''
        Initialize Activations Tensors.

        Args:
            layers: List of int describing number of neurons in each layer.
        '''
        
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations =  activations
            
   
    def sigmoid_activation(self, x):
        
        '''
        Sigmoid activation Function.

        Args:
            x: numpy.array
        Returns:
            z: numpy.array
        '''
        
        z = 1.0 / (1.0 + np.exp(-x))
        return z

    def sigmoid_derivative(self, x):
        
        '''
        Derivative of Sigmoid function.

        Args:
            x : numpy.array
        Returns:
            ds: numpy.array
        '''
        
        ds = x * (1.0 - x)
        return ds
    
    def softmax(self, x):
        
        '''
        Softmax function.

        Args:
            x : numpy.array
        Returns:
            s : numpy.array
        '''        
        
        return np.exp(x)/np.sum(np.exp(x))

  
    def forward_prop(self, inputs):
        
        '''
        Propagates forward into the Neural Network.

        Args:
            inputs : Inputs to the neural network. (numpy.array)
        Returns:
            outputs: Outputs of the neural network. (numpy.array)
        '''

        current_activation = inputs
        self.activations[0] = current_activation
        for i in range(len(self.W)):
            next_inputs = np.dot(current_activation, self.W[i]) + self.B[i]
            current_activation = self.sigmoid_activation(next_inputs) 
            self.activations[i + 1] = current_activation
        return self.softmax(current_activation)

    
    def backward_prop(self, dL):
        
        '''
        Propagates backward into the Neural Network through chain rule.

        Args:
            dL : Derivative of the Loss function with respect to output of previous layer.
        Returns:
            Updates all the derivative based on the derivative of Loss.
        '''
        
        for i in reversed(range(len(self.dW))):
            next_activation = self.activations[i + 1]
            derivative = dL * self.sigmoid_derivative(next_activation) 
            derivative_re = derivative.reshape(-1,derivative.shape[0])
            current_activation = self.activations[i].reshape(self.activations[i].shape[0],-1)
            self.dW[i] = np.dot(current_activation, derivative_re)
            self.dB[i] = derivative_re.T
            dL = np.dot(derivative, self.W[i].T)



    def update_weights(self, learning_rate, t , beta1, beta2, epsilon=1e-8): # Vanilla Updates
        
        '''
        Updates the Weights and biases based on the derivatives.

        Args:
            learning_rate: Constant multiplied to the update. Decides the step-size of the update.
        '''
        
        for i in range(len(self.W)):
            self.dW[i] = self.dW[i].reshape(self.W[i].shape)
            self.dB[i] = self.dB[i].reshape(self.B[i].shape)            
            self.M[i] = beta1 * self.M[i] + (1 - beta1) * self.dW[i]
            self.V[i] = beta2 * self.V[i] + (1 - beta2) * self.dW[i]**2
            m_hat = self.M[i] / (1 - (beta1**t))
            v_hat = self.V[i] / (1 - (beta2**t))
            self.W[i] += -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.B[i] += -learning_rate * self.dB[i]

 

    def cross_entropy_loss(self, target, output):
        
        '''
        Computes the Log-Loss.

        Args:
            target: Actual target values (numpy.array)
            output: Predicted target values (numpy array)
        Returns:
            Logloss: (float)
        '''
        
        entropy_loss = target*np.log(output)
        entropy_loss[~np.isfinite(entropy_loss)] = 0
        return -np.sum(np.sum(entropy_loss, axis=1))/target.shape[0]

    
    def GradientDescent(self, inputs, targets, epochs=1000, learning_rate=0.01, beta1=0.98, beta2=0.99):
        
        '''Performs Gradient Descent to find optimal weights and biases
           on the Logistic Loss objective function.

        Args:
            inputs       : Inputs to the neural Network (numpy.array)
            outputs      : Target to the neural Network (numpy.array)
            epochs       : Maximum iterations to repeat the gradient descent. (int)
            learning_rate: Constant multiplied to the update. Decides the step-size of the update. (float)
            beta1        : Term multiplied to the first Moment of Gradient (Adam Optimiser) (float)
            beta2        : Term multiplied to the second Moment of Gradient (Adam Optimiser) (float)
        Returns:
            Trained Multi_Layered_Perceptron object.
        '''

        for t in tqdm(range(1, epochs+1)):
            for j, Input in enumerate(inputs):
                target = targets[j]
                output = self.forward_prop(Input)
                dL = output - target
                self.backward_prop(dL)
                self.update_weights(learning_rate, (j+1)*t, beta1, beta2)
            y_pred = self.predict_proba(inputs)
            current_loss = self.cross_entropy_loss(targets, y_pred)
            self.losses.append(current_loss)


   
    def BatchSGD(self, inputs, targets, epochs=1000, learning_rate=0.01, batchsize=100, beta1=0.98, beta2=0.99):
        
        '''Performs Gradient Descent to find optimal weights and biases
           on the Logistic Loss objective function.

        Args:
            inputs       : Inputs to the neural Network (numpy.array)
            outputs      : Target to the neural Network (numpy.array)
            epochs       : Maximum iterations to repeat the mini-batch stochastic gradient descent. (int)
            learning_rate: Constant multiplied to the update. Decides the step-size of the update. (float)
            batchsize    : Number of samples to consider for computing the derivative and Weights and Biases. (int)
            beta1        : Term multiplied to the first Moment of Gradient (Adam Optimiser) (float)
            beta2        : Term multiplied to the second Moment of Gradient (Adam Optimiser) (float)
        Returns:
            Trained Multi_Layered_Perceptron object.
        '''

        input_batches = np.array_split(inputs, inputs.shape[0]//batchsize)
        target_batches = np.array_split(targets, inputs.shape[0]//batchsize)
        for t in tqdm(range(1, epochs+1)):
            for i in range(len(input_batches)):
                dL=0
                for j, Input in enumerate(input_batches[i]):
                    target = target_batches[i][j]
                    output = self.forward_prop(Input)
                    dL += (output - target)/batchsize
                self.backward_prop(dL)
                self.update_weights(learning_rate, t, beta1, beta2)
            y_pred = self.predict_proba(inputs)
            current_loss = self.cross_entropy_loss(targets, y_pred)
            self.losses.append(current_loss)


    def predict_proba(self, Inputs):
        
        '''
        Propagates forward in a trained neural netwrok to predict probability outputs.

        Args:
            inputs     : Inputs to the Multi Layered Perceptron model. (numpy.array)
        Returns:
            predictions: Outputs of the Multi Layered Perceptron model. (numpy.array)
        '''
        
        predictions = np.array([self.forward_prop(Input) for Input in Inputs])
        return predictions

   
    def predict(self, Inputs):
        
        '''
        Propagates forward in a trained neural netwrok to predict class outputs.

        Args:
            inputs     : Inputs to the Multi Layered Perceptron model. (numpy.array)
        Returns:
            predictions: Outputs of the Multi Layered Perceptron model converted to a class labels. (numpy.array)
        '''
        pred = []
        y_pred = self.predict_proba(Inputs)
        for y in y_pred:
            p = np.zeros_like(y)
            p[np.argmax(y)] = 1
            pred.append(p)
        return np.array(pred)
    
    def plot_loss(self):    
        
        '''
        Plots Epoch vs Loss graph for a trained Multi Layered Perceptron Model.

        Args:
            Multi_Layered_Perceptron objetc.
        Returns:
        '''
        
        ep = list(range(1, len(self.losses)+1))
        plt.xlim(0, len(self.losses)+1)
        plt.plot(ep, self.losses, 'b-')
        plt.title('Epochs vs Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


    
    
if __name__ == "__main__":
    

    # Benchmarking the perfromance of our MLP on MNIST dataset.
    print('Preprocessing the MNIST Data...')
    df_train, X_train, Y_train = preprocess_data_mnist('mnist_train.csv')
    df_test, X_test, Y_test = preprocess_data_mnist('mnist_test.csv')
    print('Normalizing the MNIST Data...')
    X_train, train_means, train_stds = normalize_train_data(X_train)
    X_test = normalize_test_data(X_test, train_means, train_stds)


    # create a Multilayer Perceptron with default parameters
    print('Creating the architecture for Neural Network...')
    inputs = 784; layers = [32, 16]; output = 10
    mlp = Multi_Layered_Perceptron(n_inputs=inputs, hidden_layers=layers, n_outputs=output, activation_function='sigmoid')

    # training the network
    print('Training the Neural Network...')
    mlp.GradientDescent(X_train.values[:10000], Y_train.values[:10000], learning_rate=0.01, epochs=100,  beta1=0.9, beta2=0.999)
    mlp.plot_loss()

    # Evaluate Performance
    print("Performance Evaluation")
    y_pred_train_classes = mlp.predict(X_train.values[:10000])
    y_pred_test_classes = mlp.predict(X_test.values)
    print('Training accuracy \t:',  round(accuracy(Y_train.values, y_pred_train_classes), 4))
    print('Testing accuracy \t:', round(accuracy(Y_test.values, y_pred_test_classes), 4))
