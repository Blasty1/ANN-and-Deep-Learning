from util import *
from tqdm import tqdm

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        self.weight_decay=0.0002
        
        self.weight_h_to_v = None

        self.learning_rate = 0.001
        
        self.momentum = 0.7

        self.print_period = 5000
        
        self.take_reconstruction_error_period = 100
        
        self.recon_errors = []
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations):
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            visible_batch = visible_trainset[batch_indices]

	    # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
            p_h_given_v , h0 = self.get_h_given_v(visible_batch) # v_0 -> h_0
            p_v_given_h , v1 = self.get_v_given_h(h0) # h_0 -> v_1
                       
            p_h_given_v_recon, h1 = self.get_h_given_v(v1) # v_1 -> h_1

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            self.update_params(visible_batch,h0,v1,h1)
            
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:

                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # Every 100 iterations
            if it % self.take_reconstruction_error_period == 0: 
                reconstruction_loss = np.mean((visible_batch - v1) ** 2)
                self.recon_errors.append((it,reconstruction_loss))
        
            
            # print progress
            if it % self.print_period == 0 :
                reconstruction_loss = np.mean((visible_batch - v1)**2)
                print ("iteration=%7d recon_loss=%4.4f"%(it,reconstruction_loss))
        
        return
    
    def cd1_batch(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling
            with mini-batch 
        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        batch_size = self.batch_size
        n_batches = n_samples // batch_size


        for it in range(n_iterations):
            indices = np.random.permutation(n_samples)
            shuffled_data = visible_trainset[indices]
            for batch_idx in tqdm(range(n_batches)):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                v_batch = shuffled_data[batch_start:batch_end]

	    # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
                p_h_given_v , h0 = self.get_h_given_v(v_batch) # v_0 -> h_0
                p_v_given_h , v1 = self.get_v_given_h(h0) # h_0 -> v_1
                       
                p_h_given_v_recon, h1 = self.get_h_given_v(v1) # v_1 -> h_1

            # [TODO TASK 4.1] update the parameters using function 'update_params'
                self.update_params(v_batch,h0,v1,h1)
            
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:

                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # Every 100 iterations
            # if it % self.take_reconstruction_error_period == 0: 
            #     reconstruction_loss = np.mean((visible_batch - v1) ** 2)
            #     self.recon_errors.append((it,reconstruction_loss))
        
            
            # print progress
            if it % self.print_period == 0 :

                print ("iteration=%7d recon_loss=%4.4f"%(it, np.linalg.norm(v_batch - v1)))
                with open('dbm_log.txt', 'a') as f:
                    f.write(f"{it} {np.linalg.norm(v_batch - v1)}\n")
        
        return

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        batch_size = v_0.shape[0]

        
        # Compute gradients
        grad_bias_v = np.sum(v_0 - v_k, axis=0) / batch_size
        grad_weight_vh = (v_0.T @ h_0 - v_k.T @ h_k) / batch_size
        grad_bias_h = np.sum(h_0 - h_k, axis=0) / batch_size
        
        # Apply weight decay to weights only (not biases)
        grad_weight_vh = grad_weight_vh - self.weight_decay * self.weight_vh
        
        # Momentum
        self.delta_bias_v = self.momentum * self.delta_bias_v + grad_bias_v
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + grad_weight_vh
        self.delta_bias_h = self.momentum * self.delta_bias_h + grad_bias_h
        
        # Update
        self.bias_v += self.learning_rate * self.delta_bias_v
        self.weight_vh += self.learning_rate * self.delta_weight_vh
        self.bias_h += self.learning_rate * self.delta_bias_h
        
        
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        support = self.bias_h + visible_minibatch @ self.weight_vh

        #P(H|V)
        p_h_given_v_value = sigmoid(support)
        
        #activations h
        #samples from a Bernoulli distribution
        h_sample = sample_binary(p_h_given_v_value)

        
        return p_h_given_v_value, h_sample


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            
            # Compute total input to visible layer
            support = self.bias_v + hidden_minibatch @ self.weight_vh.T
            
            # Split into data part and label part
            data_support = support[:, :-self.n_labels]
            label_support = support[:, -self.n_labels:]
            
            # Data part: sigmoid activation + binary sampling
            p_v_data = sigmoid(data_support)
            v_data = sample_binary(p_v_data)
            
            # Label part: softmax activation + categorical sampling
            p_v_label = softmax(label_support)
            v_label = sample_categorical(p_v_label)
            
            # Concatenate back together
            p_v_given_h_value = np.concatenate([p_v_data, p_v_label], axis=1)
            v_sample = np.concatenate([v_data, v_label], axis=1)
            
            return p_v_given_h_value, v_sample
        else:
                        
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             
           
           # Compute support (input to visible layer)
            support = self.bias_v + hidden_minibatch @ self.weight_vh.T

            # P(V|H) using sigmoid activation
            p_v_given_h_value = sigmoid(support)

            # Sample from Bernoulli distribution
            v_sample = sample_binary(p_v_given_h_value)

            return p_v_given_h_value, v_sample

    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def get_riconstruction_error(self):
        return self.recon_errors

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # // [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        support = self.bias_h + visible_minibatch @ self.weight_v_to_h

        #P(H|V)
        p_h_given_v_value = sigmoid(support)
        
        #activations h
        #samples from a Bernoulli distribution
        h_sample = sample_binary(p_h_given_v_value)

        
        return p_h_given_v_value, h_sample
        


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # // [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            raise RuntimeError("get_v_given_h_dir called on top RBM; top stays undirected in a DBN.")
            
        else:
                        
            # // [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            # Compute support (input to visible layer)
            support = self.bias_v + hidden_minibatch @ self.weight_h_to_v #! i delete it: .T

            # P(V|H) using sigmoid activation
            p_v_given_h_value = sigmoid(support)

            # Sample from Bernoulli distribution
            v_sample = sample_binary(p_v_given_h_value)

            return p_v_given_h_value, v_sample
            
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v  
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
