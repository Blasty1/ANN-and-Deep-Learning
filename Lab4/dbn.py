from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy
    
        Args:
          true_img: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
         # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        n_samples = true_img.shape[0] 
        batch_size =self.batch_size
        all_preds = []
        correct = 0

        for i in range(0, n_samples, batch_size):
            vis_batch = true_img[i:i+batch_size]
            lbl_batch = true_lbl[i:i+batch_size]
            lbl = np.ones(lbl_batch.shape) / float(self.sizes["lbl"])

            p_hid, _ = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_batch)
            p_pen, _ = self.rbm_stack["hid--pen"].get_h_given_v_dir(p_hid)
            pen = p_pen

            rbm_top = self.rbm_stack["pen+lbl--top"]
            n_pen = self.sizes["pen"]
            top_v = np.concatenate((pen, lbl), axis=1)

            for _ in range(self.n_gibbs_recog):
                p_h, top_h = rbm_top.get_h_given_v(top_v)
                p_v, top_v_sample = rbm_top.get_v_given_h(top_h)
                top_v = np.concatenate((pen, p_v[:, n_pen:]), axis=1)

            # predicted_lbl = np.zeros(true_lbl.shape)
            predicted_lbl = top_v[:, n_pen:] # shape (n_samples, n_labels)
            all_preds.append(predicted_lbl)
            correct += np.sum(np.argmax(predicted_lbl, axis=1) == np.argmax(lbl_batch, axis=1))

        predicted_lbl_full = np.vstack(all_preds)
        accuracy = 100.0 * correct / n_samples

        print("accuracy = %.2f%%" % accuracy)
        return predicted_lbl_full, accuracy

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
       
        n_sample = true_lbl.shape[0]
        batch_size = self.batch_size
        rbm_top = self.rbm_stack["pen+lbl--top"]
        n_pen = self.sizes["pen"]
        generated_images = []
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])
        single_lbl_index = np.argmax(true_lbl[0])

        for i in range(0, n_sample, batch_size):
            lbl_batch = true_lbl[i:i+batch_size]
            current_batch = lbl_batch.shape[0]

            pen = np.random.rand(current_batch, n_pen)
            top_v = np.concatenate((pen, lbl_batch), axis=1)

            for _ in range(self.n_gibbs_gener):
                p_h, top_h = rbm_top.get_h_given_v(top_v)
                p_v, top_v_sample = rbm_top.get_v_given_h(top_h)
                top_v = np.concatenate((p_v[:, :n_pen], lbl_batch), axis=1)

                pen_current = top_v[:, :n_pen]

                p_hid, hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_current)
                p_vis, vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid)
                if i == 0: # record only the first batch
                    vis_sample_0 = p_vis[0].reshape(self.image_size)
                    records.append( [ ax.imshow(vis_sample_0, cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name, single_lbl_index))            


        return generated_images

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # // [TODO TASK 4.2] use CD-1 to train all RBMs greedily
            
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """ 
            self.rbm_stack['vis--hid'].cd1(vis_trainset)

            p_h_given_v_vis, h_sample_vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_trainset)
            hid_trainset = p_h_given_v_vis

            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")


            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """            
            self.rbm_stack["vis--hid"].untwine_weights() 

            self.rbm_stack["hid--pen"].cd1(hid_trainset)    

            p_h_given_v_hid, h_sample_hid = self.rbm_stack["hid--pen"].get_h_given_v(hid_trainset)
            pen_trainset = p_h_given_v_hid

            self.savetofile_rbm(loc="trained_rbm",name="hid--pen") 

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()

            pen_plus_lbl = np.concatenate([pen_trainset, lbl_trainset], axis=1)
            self.rbm_stack["hid--pen"].cd1(pen_plus_lbl)
            
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")          

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
