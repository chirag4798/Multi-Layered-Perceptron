from utility_functions import *

class Multi_Layered_Perceptron:
    
    '''
    A Multi Layered Perceptron model.

    Attributes:
        n_inputs     : Number of neurons in the input layer (int; default=8).
        hidden_layers: Number of neurons in each hidden layers (list of int; default=[8, 6, 4]).
        n_outputs    : Number of neurons in the output layer (int; default=2).
        random_state : Value given to np.random.seed() (int; default=1).
    Returns:
        Initializes a Multi Layered Perceptron model with
        sigmoid activation function and LogLoss as its objective function.
    Methods:
        1. GradientDescent(self, X, y)
        2. BatchSGD(self, X, y)
        3. predict_proba(self, X)
        4. preict(self, X)
        5. plot_loss(self)
    '''
    
    def __init__(self, n_inputs=8, hidden_layers=[8, 6, 4], n_outputs=2, random_state=1): # Default values suitable for our problem

        '''
        Constructer for the Multi_Layered_Perceptron object.
        '''
        
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
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
        W, dW, B, dB = [],[],[],[]    
        for i in range(1,len(layers)):
            np.random.seed(self.random_state)
            w, dw = np.random.rand(layers[i-1], layers[i]), np.zeros((layers[i-1], layers[i]))
            b, db = np.random.rand(layers[i]), np.zeros(layers[i])
            W.append(w)
            dW.append(dw)
            B.append(b)
            dB.append(db)

        self.W, self.dW, self.B, self.dB =  W, dW, B, dB
        
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
        for i, parameter in enumerate(zip(self.W, self.B)):
            weights = parameter[0]
            bias = parameter[1]
            next_inputs = np.dot(current_activation, weights) + bias
            current_activation = self.sigmoid_activation(next_inputs)
            self.activations[i + 1] = current_activation
        return current_activation

    
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
            current_activation = self.activations[i]
            self.dW[i] = np.dot(current_activation.reshape(current_activation.shape[0],-1), derivative.reshape(-1,derivative.shape[0]))
            self.dB[i] = derivative.reshape(derivative.shape[0], -1)
            dL = np.dot(derivative, self.W[i].T)



    def update_weights(self, learning_rate): # Vanilla Updates
        
        '''
        Updates the Weights and biases based on the derivatives.

        Args:
            learning_rate: Constant multiplied to the update. Decides the step-size of the update.
        '''
        
        for i in range(len(self.W)):
            self.dW[i] = self.dW[i].reshape(self.W[i].shape)
            self.dB[i] = self.dB[i].reshape(self.B[i].shape)
            self.W[i] += -learning_rate * self.dW[i]
            self.B[i] += -learning_rate * self.dB[i]

 

    def logloss(self, target, output):
        
        '''
        Computes the Log-Loss.

        Args:
            target: Actual target values (numpy.array)
            output: Predicted target values (numpy array)
        Returns:
            Logloss: (float)
        '''
        
        log_loss = target*np.log(output) + (1 - target)*np.log(1 - output) 
        return -np.mean(log_loss)

    
    def GradientDescent(self, inputs, targets, epochs=1000, learning_rate=0.01, tolerance=1e-4):
        
        '''Performs Gradient Descent to find optimal weights and biases
           on the Logistic Loss objective function.

        Args:
            inputs       : Inputs to the neural Network (numpy.array)
            outputs      : Target to the neural Network (numpy.array)
            epochs       : Maximum iterations to repeat the gradient descent. (int)
            learning_rate: Constant multiplied to the update. Decides the step-size of the update. (float)
            tolerance    : Maximum accepted deviation in the Loss in one epoch. (float)
        Returns:
            Trained Multi_Layered_Perceptron object.
        '''

        prev_loss = 10
        for t in tqdm(range(epochs)):
            for j, Input in enumerate(inputs):
                target = targets[j]
                output = self.forward_prop(Input)
                dL = (output - target)/(output*(1 - output)) # dL/da for Logloss
                self.backward_prop(dL)
                self.update_weights(learning_rate)
            y_pred = self.predict_proba(inputs)
            current_loss = self.logloss(targets, y_pred)
            self.losses.append(current_loss)
            
            if current_loss > prev_loss + tolerance:
                print('Early Stopping!')
                break
            else:
                prev_loss = current_loss 
                continue


   
    def BatchSGD(self, inputs, targets, epochs=1000, learning_rate=0.01, batchsize=5, tolerance=1e-4):
        
        '''Performs Gradient Descent to find optimal weights and biases
           on the Logistic Loss objective function.

        Args:
            inputs       : Inputs to the neural Network (numpy.array)
            outputs      : Target to the neural Network (numpy.array)
            epochs       : Maximum iterations to repeat the mini-batch stochastic gradient descent. (int)
            learning_rate: Constant multiplied to the update. Decides the step-size of the update. (float)
            batchsize    : Number of samples to consider for computing the derivative and Weights and Biases. (int)
            tolerance    : Maximum accepted deviation in the Loss in one epoch. (float)
        Returns:
            Trained Multi_Layered_Perceptron object.
        '''

        input_batches = np.array_split(inputs, inputs.shape[0]//batchsize)
        target_batches = np.array_split(targets, inputs.shape[0]//batchsize)
        prev_loss = 10
        for t in tqdm(range(epochs)):
            for i in range(len(input_batches)):
                dL = 0
                for j, Input in enumerate(input_batches[i]):
                    target = target_batches[i][j]
                    output = self.forward_prop(Input)
                    dL += ((output - target)/(output * (1 - output)))/(batchsize) 
                self.backward_prop(dL)
            self.update_weights(learning_rate)
            y_pred = self.predict_proba(inputs)
            current_loss = self.logloss(targets, y_pred)
            self.losses.append(current_loss)
            
            if current_loss > prev_loss + tolerance:
                print('Early Stopping!')
                break
            else:
                prev_loss = current_loss 
                continue



    def predict_proba(self, Inputs):
        
        '''
        Propagates forward in a trained neural netwrok to predict probability outputs.

        Args:
            inputs     : Inputs to the Multi Layered Perceptron model. (numpy.array)
        Returns:
            predictions: Outputs of the Multi Layered Perceptron model. (numpy.array)
        '''
        
        predictions = []
        for Input in Inputs:
            p = self.forward_prop(Input)
            predictions.append(p)
        return np.array(predictions)

   
    def predict(self, Inputs):
        
        '''
        Propagates forward in a trained neural netwrok to predict class outputs.

        Args:
            inputs     : Inputs to the Multi Layered Perceptron model. (numpy.array)
        Returns:
            predictions: Outputs of the Multi Layered Perceptron model converted to a class labels. (numpy.array)
        '''
        
        y_pred = self.predict_proba(Inputs)
        pred = []
        for i,j in y_pred:
            if i > j:
                pred.append(0)
            else:
                pred.append(1)
        return np.array(pred)
    
    def plot_loss(self):    
        
        '''
        Plots Epoch vs Loss graph for a trained Multi Layered Perceptron Model.

        Args:
            Multi_Layered_Perceptron objetc.
        Returns:
        '''
        
        ep = list(range(1, len(self.losses)+1))
        plt.xlim(1, len(self.losses)+1)
        plt.plot(ep, self.losses, 'b-')
        plt.title('Epochs vs Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.grid()
        plt.show()
    
    
if __name__ == "__main__":
    

    # Benchmarking the perfromance of our MLP on Titanic Dataset from Kaggle.
    df, X, Y = preprocess_train_data('titanic_train.csv')
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_train, train_means, train_stds = normalize_train_data(X_train)
    X_test = normalize_test_data(X_test, train_means, train_stds)

    # create a Multilayer Perceptron with default parameters
    mlp = Multi_Layered_Perceptron(random_state=8)

    # training the network
    mlp.GradientDescent(X_train.values, Y_train.values, epochs=1000)
    mlp.plot_loss()

    # Evaluate Performance
    print("Performance Evaluation")
    y_pred_train_classes = mlp.predict(X_train.values)
    y_pred_test_classes = mlp.predict(X_test.values)
    print_classification_report(Y_train.values[:,1], Y_test.values[:,1], y_pred_train_classes, y_pred_test_classes) 
    plot_confusion_matrix(Y_train.values[:,1], y_pred_train_classes, 'Train')
    plot_confusion_matrix(Y_test.values[:,1], y_pred_test_classes, 'Test')
    plot_ROC_curve(X_train, Y_train, X_test, Y_test, mlp)







