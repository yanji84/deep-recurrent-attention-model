This is my attempt to replicate the results of this paper (https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf) using Tensorflow. I have found the torch implementation here (http://torch.ch/blog/2015/09/21/rmva.html) very helpful

Current image module in Tensorflow does not support padding/cropping using tensor. This has forced me to run computation graph evaluation at each step of the main RNN. I am not quite satisfied with the current design. Still looking for ways to get around this in Tensorflow
