class pcn_logic: # implementation of logical perceptron
    
    def __init__(self):
        pass
    
    def train(self, inputs, targets, lr, n_iter):
        
        n_obs = np.shape(inputs)[0]
        n_in = np.shape(inputs)[1]
        n_out = np.shape(targets)[1]
        
        inputs = np.concatenate((inputs, - np.ones((n_obs,1))), axis = 1)
        
        weights = np.random.rand(n_in + 1,n_out)*0.1 - 0.05
        
        
        for i in range(n_iter):
            print('Iteration: {}\n'.format(i + 1))
            
            activation = np.where(np.dot(inputs, weights) > 0, 1, 0)
            
            print('Current Result\n{}\n'.format(activation))
            print('Target\n{}'.format(targets))
            print('-----------------------')
            
            if np.array_equal(activation,targets):
                break
            
            else:
                weights -= lr*np.dot(np.transpose(inputs),(activation - targets))
            
        self.weights = weights
        print('\npcn train completed')