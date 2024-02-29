# nn-zero-to-hero-notes
Learning notes on Andrej Karpathy's course, [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero?tab=readme-ov-file).

Big shout out to Andrej, one of the greatest ML educators of our times. Highly recommend to check out his Github repos and Youtube videos.

## 1. Building micrograd
1. Build a tiny autograd engine implementing backpropagation using pure Python, with cool visualization.
2. Compare with PyTorch's implementation.
3. Build a small neural networks library on top of micrograd with PyTorch-like APIs.


## 2. Building makemore Part 1: bigram and simple NN
Makemore is a character-level language model that takes a text file as input and outputs more things like it, implemented by creating a bigram model and a simple NN this time.
1. bigram
- 1.1. Create N, the 'model'
- 1.2. Trainning:
    - For efficiency, create P as a matrix of prob,use /= instead of P = P/, because it's inplace so more efficient.
- 1.3. Sampling/inference
- 1.4. loss func
    - Goal: maximize likelihood of the data w.r.t. model parameters (statistical modeling);<br>
        - equivalent to maximizing the log likelihood, because log is monotonic;<br>
        - equivalent to minimizing the negative log likelihood <br>
        - equivalent to minimizing the average negative log likelihood <br>
    log(a*b*c) = log(a) + log(b) + log(c)

    - Use negative log likelihood with loss smoothing, +1 (in P = (N+1).float())for smoothing the loss func to cover some extreme cases.<br>
    `P = (N+1).float()`
2. Simple NN 
- 2.1. Create datasets
    - Preferably use torch.tensor instead of torch.Tensor, as Tensor set dtype to the default float32
    - use .float() to convert to float32 as F.one_hot only returns the same dtype of input
- 2.2. Initialize the 'network'
- 2.3. Gradient descent
    - 2.3.1. Forward pass
    - 2.3.2. Backward pass
    - 2.3.3. Update

## 3. Building makemore Part 2: MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Use .view to change the matrix shape, because it's much moe efficient. See [this blog post on PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/).
- Steps to determine the initial learning rate:
    - 1. Manually set a wide range of rates to get a sense, 0.001 ~ 1 looks good
    - 2. Generate a thousand rates within the range, but exponentially
    - 3. Run the optimization for 1000 steps, using the learning rates indexed, and keep tracking of the loss
    - 4. Plot the stats to find a good spot for a learning rate
- Training, dev/val, test split
    - Test split should be rarely used at the end, because if it's used a lot, as every time we learn something from it, it's equivelent to train on the test split

## 4. Building makemore Part 3: Activations & Gradients, BatchNorm

- Good practice to plot paramters to check if things are well calibrated. E.g., if there are dead neurons.

- About batch norm:
    - Intuition: we want the hidden states hpreact (hidden layer pre-activation) to be roughly gaussian. Because if too small then tanh won't be doing anything, if too large then tanh will be too saturated. And the calculations are perfectly differentialble.
    - So we normalize hpreact. And we want hpreact to be roughly gaussian only at init, we don't want to force it to be gaussian always.We like the distribution to be able to move around, told be the backprop.
    - So we use scale and shift. Init scale (bngain) as 1s and shift (bnbias) as 0s, so that in the init it's gaussian and during optimization we'll be able to backprop the bngain and bnbias so the network has the full ablity to do whatever it wants.

    - In large network, it's better and easier to 'sprinkle' batch norm layers throughout the network. It's common to append a batch norm layer right after a linear layer or convolutional layer, to control these activations at every point in the network.

    - Good side effect comes from batch norm: regularization. Each hpreact will change a little depend on the mean and std of that batch, so the h will change s a little coresspondingly. The change of h introduces some entropy/noise (also like a form of data augmentation), equivelent to regularization, which is a good thing in network training.

    - People trying to use other norm techniques that do not couple the examples in a batch, eg. layer norm, instance norm, group norm, etc., but it's been hard to remove batch norm because it works so well due to its regularizing effect.

    - In the test/inference stage, how are we gonna get prediction of a single input if the model is expecting batch inputs? 
        - One way to solve this is to add one more step to calculate the mean and std of the batch norm over the entire training set (called calibrating the batch norm stat).
        - Another (perferred, and implememted in pytorch) way is to estimate the bnmean and bnstd during training in a running mean manner, so that we don't need a separate stage for calculating the bnmean and bnstd after training.

    - Adding batch norm layer makes it less required to choose the right gain, too low or too high gain resulting the similar good distruibutions of activation, grad and weights grad, because batch norm makes training more robust. But the scale of update-to-data is effected if gain is set too high or too low, so we can adjust the learning rate accordingly.

    - Batch norm makes training more robust, even if we forgot / fan_in**0.5 when init the weights and set gain to 1 (no gain), we can still get good results. We just need to adjust (usually larger) the learning rate according the update_to_data plot.

- About kaiminng normalization:
    - We want the std of hpreact( = embcat @ W1 + b1, ignoring b1 since we set it to very small ) to be 1, so we need to mutiply W1 with a small number. 
    - For tanh, it's (5/3)/((fan_in)**0.5). Different nonlinearilty requires diff factor number, check torch.nn.init.kaiming_normal_()
    - kaiming_normal_ is the most popular way of init
    - We can also init the grad(mode='fan_out'), instead of activation(mode='fan_in'), choose one of them as these two don't diff too much.
    - Back in 2015 when the kaiminng normal paper published, we had to be extremely careful with the activations and grads, especially when the nn is very deep.
- Later a number of modern innovations (e.g. 
    1. residual connections; 
    2. batch norm, layer norm, group norm; 
    3. better optimizers: adam, RMSprop) 

    make things more stable and well-behaved, so it became less imortant to init the networks "extactly right".

## 5. Building makemore Part 4: becoming a backprop ninja
skipped for now.

## 6. Building makemore Part 5: Building a WaveNet, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499).
- We only look at training loss in this exercise, normally we look at both the tarining and validation loss together.
- We only implement the hierarchical architecture of the wavenet, but not the complicated forward pass (with residual and skip connections) in the paper.
- Brief convolutions preview/hint: this use of convolutions is strictly for efficiency. It does not change the model we implemented here. Efficiency: 1) the for loop is not outside in python, but inside of the kernals in CUDA; 2) values in nodes are reused for each 'linear filter'.
- We pytorchified the code here. pytorch is better, but sometimes the documentation is a bit messy
- It's a good practice that prototype in jupyter notebook and make sure that all the shapes work out first and then copy the code to the repository to do actual training/experiments.
- The matrix multiplication operator @ in PyTorch only works on the last dimension. torch.randn(4, 3, 29, 80) @ torch.randn(80, 200) gets a size of ([4, 3, 29, 200])


## 7. Let's build GPT

- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights. In principle it can be applied to any arbitatry directed graph.
- There is no notion of space. Attention simply acts over a set of vectors.  This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other. So in the above example, there 4 groups of 8 nodes processing independently.
- In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides `wei` by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. The scaling is used just to control the variance at init. 

- Self-attention is for the tokens to do communication, once they have gathered the information, they need to 'think' on that information individually, so feedfoward layer right after self-attention layer. 
- Communication using attention, followed by computation using feedfoward,on all the tokens independently. We use block to intersperse communication and computation.
- Deep neural nets suffer from optimization issues. 2 optimizations taht dramatically help with the depth of networks and make sure that the networks remian optimizable: 
    1.  skip connections/residual connections ([kaimin et al 2015](https://arxiv.org/abs/1512.03385)): 
        - Add the input that before the computation of this layer: `x = x + self.ffwd(x)`
        - This is useful because addition distributes gradients equally to both of its branches. 
        - The residual blocks are initialized in the beginning so they contribute very little, if anything, to the residual pathway, but during optimization they 'come online' over time and start to contribute.
    2. layer norm ([Geoffrey E. Hinton et al 2016](https://arxiv.org/abs/1607.06450))
        - Similar to batch norm
        - Where to place it: one of the very few changes people now commonly used that slightly departs from the original transformer paper. In the original paper, layer norm is applied after the transfotmation. Now it's more common to apply the layer norm before the transfotmation, which is called pre-norm formulation.
        - Kind of like a per token transfotmation that normalizes the features and makes them unit gaussian at initialization. But of course because there are trainable parameters gamma and beta inside the layer norm, the layer norm eventually create outputs that might not be unit gaussian, determined by optimization.
- Dropout ([Hinton et al 2014](https://dl.acm.org/doi/pdf/10.5555/2627435.2670313)):
    - A regularization technique: randomly drop some neurons to zero in every forward and backward path, and train without them. It's like training an ensemble of sub networks. During test time everything is fully enabled so all of the sub networks are merged into an ensemble.
    - Where to place it:
        1. Can add dropout layer at the end of the FeedFoward right before going back into the residual pathway.
        2. Can add dropout layer at the end of the MultiHeadAttention right after the projection layer before going back into the residual pathway.
        3. Can add dropout layer when calculating the "affinities" in the Head after the softmax layer. It randomly prevent some nodes from communicating after the softmax. 

- Some hyperparameters:
    - `n_embd = 384`, `n_head = 6`, so every head is 384/6=64 dimensional as a standard.
- nanoGPT: model is almost identical to the one in this tutorial, except some parts, e.g.:
    1. using gelu, same as GPT2
    2. separate n_head as 4th dimension, so that all the heads are treated as batch dimenison as well (inside CausalSelfAttention), which is more efficient.
    3. etc
    4. Check out the train.py for techniques in  weigjht decay and saving and loading checkpoints.

- Notes for the python script:
    - Device option: cuda, cpu. Make sure to move the input data and model to the device, and generate outputs from the device.
    - eval_itrs and estimate_loss func: the loss of each batch is noisy because each batch can be more or less lucky. So estimate_loss averages losses of multiple batches, gives a more accurate measurement of the current loss.
    - make sure the correct mode of the model, i.e., training mode or eval(inference) mode, because some layers (e.g., dropout layes, batch norm layers, etc.) will have different behaviors at training or inference time.
    - call context manager torch.no_grad() when grads will not be used, so it will be more memory efficient.