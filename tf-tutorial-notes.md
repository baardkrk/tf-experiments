[Tensorflow tutorial][1] Notes 
=============================

```python
const = tf.constant(2.0, name="const") # constant

b = tf.Variable(1.0, name='b')         # variable
c = tf.Variable(2.0, name='c')	       # variable
```
First parameters are the initial values, second is an optional name for the
variable/constant -- handy for visualizations. Also, notice that we use '-marks for single 
character variable names, and "-marks for strings.

Tensorflow operations:
```python
d = tf.add(b, c, name='d')          # addition
e = tf.add(c, const, name='e')	    # addition
a = tf.multiply(d, e, name='a')	    # multiplication
```

Tensorflow needs an object to initialize the variables end the graph structure.
```python
init_op = tf.global_variables_initializer()
```

For tensorflow to run, we need to create an object, `tf.Session()` where all operations
are run. This is because Tensorflow was created on the *static* graph paradigm. Another
way is [eager][2], where the graph structure are built on the fly.
```python
with tf.Session() as sess:

    sess.run(init_op)			     # initialize the variables
    a_out = sess.run(a)			     # compute the output graph

    print("Variable a is {}".format(a_out))
```
Here, we only run the `a = tf.multiply(d, e, name='a')` *operation*. The result from this
operation is assigned to `a_out` and printed. The operations `d` and `e` is run
automagically in the background.


Tensorflow Placeholders
-----------------------
If we don't know what the value of some arrays should be during the declaration phase
(before we run `with tf.Session as sess:`) we can declare their basic structure in a
placeholder:
```python
b = tf.placeholder(tf.float32, [None, 1], name='b')
```
First argument is the data *type*. Second comes the shape of the data which will be
"injected" into the variable. Here it is defined as a (? x 1) array. The 'None' parameter
is accepted in this case which makes us able to feed in as much one-dimensional data into
the b-parameter as we want when we run the graph.

The other change we need to make to our program is the `sess.run(a,...)`.

```python
a_out = sess.run(a, feed_dict={b: np.arrange(0, 10)[:, np.newaxis]})
```
The `feed_dict` argument specifies what the variable b is going to be, a one dimensional
range from 0 to 10. `feed_dict` is a python dictionary where each name (in this case only
*b*) is the key for each placeholder we are filling.


Basic MNIST neural network example
----------------------------------

MNIST is a dataset of 28x28 pixel grayscale images with hand-written digits, and the
loader for the dataset. It has 55~000 training rows, 10~000 testing rows and 5~000
validation rows.

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

`one_hot=True` lets the label associated with the image be a vector [0,0,0,0,1,0,0,0,0,0]
instead of the digit it self ("4" in this case). This lets us feed the labels more easily
into the output layer of the neural network.

```python
# setup hyperparameters and such

learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 digits
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])
```

Notice that we've "stretched" the 28x28 image into a one-dimensional vector for this
example. We lose some spacial information in this process, but more on this in
convolutional neural networks.

In this network there are three layers: input, hidden and output. We always need to set up
L-1 number of weight/bias tensors, there L is the number of layers in the network.

```python
# Declare weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# weights connecting the hidden layer to the output layer:
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')
```

We've declared 300 nodes in the hidden layer in this case. The weights are initialized
with a random normal distribution centered around zero, with a standard deviation of 0.03.

Next, we set up the node inputs and activation functions for the hidden layer nodes

```python
# calculate output for the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)  # matrix multiplication of x and W1, add b1
hidden_out = tf.nn.relu(hidden_out)   	   # applying relu on the result

# setup output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))    # using softmax activation
```

We'll also have to include a cost-function for the backpropogation:

\[
J = -\frac{1}{m} \sum\limits_{i=1}^{m}\sum\limits_{j=1}^{n} y_j^{(i)} \log(y_j\_^{(i)}) +
(1 - y_j^{(i)}) \log(1 - y_j\_^{(i)})
\]

Where $y_j^{(i)}$ is the i-th training label for output node j, y_j\_^{(i)} is the i-th
predicted label for output node j, m is the number of training/batch samples and n is the
number(?) 
We have two operations occuring:
  1) Summation of the logarithmic products, and additions *across all the ouptut nodes*.
  2) Taking the mean of this summation *across all training samples*. (in this batch?)

```python
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + 
	      					(1 - y) * tf.log(1 - y_clipped), 
						axis=1))
```
Explanation:
First, we convert the output, y_, to a clipped version, limited between 1e-10 and
0.9999999. This is to make sure we never have a case where we have log(0) during
training -- it would return NaN and break the training process.
The second line is the cross-entropy calculation. `reduce_sum` takes the sum of a given
axis in the tensor you supply.
In this case, the tensor supplied is the element-wise cross-entropy calculation for a
single node and training sample. i.e. $y_j^{(i)} \log(y_j\_^{(i)}) + (1 - y_j^{(i)})
\log(1 - y_j\_^{(i)})$, or `y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped)`.
`tf.reduce_mean` takes us the last step by taking the mean of whatever tensor we provide
it with.

Now, we set up the optimizer
```python
optimiser =
tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
```

We use the built in gradient descent optimizer. It needs to be initialized with a learning
rate, and that we specified what we want it to do. In this case to minimize the cross
entropy cost function we created for the backpropogation.
A library of popular nn training optimizers can be found [here][3].

Finally, we set up the variable initialization operation and an operation to measure the
accuracy of our results:
```python
init_op = tf.global_variables_initializer()  # initialization operation

# define an accuracy assesment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
This^^ is 'self-documenting'

Now, we just have to set up the training
```python
with tf.Session() as sess:
     # initialize the variables
     sess.run(init_op)
     total_batch = int(len(mnist.train.labels) / batch_size)
     for epoch in range(epochs):
     	 avg_cost = 0
	 for i in range(total_batch):
	     batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
	     _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y}) 
	     avg_cost += c / total_batch
	 print("Epoch: ", (epoch +1), " cost = ", "{:.3f}".format(avg_cost))
     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
```

+-------------------------------------------------------------------------------------+
| Some [terminology][4]								      |
+----------------+--------------------------------------------------------------------+ 
| Loss function: | Defined on on a data point, prediction and label, and measures the |
|                | penalty.                                                           |
| 		 | Square loss: l( f(x_i | \theta), y_i) = (f(x_i | \theta) - y_i)^2  |
|		 | 	  	used in linear regression. 	    	      	      |
|		 | Hinge loss: l( f(x_i | \theta), y_i) = max(0, 1 - f(x_i | \theta)) |
|		 | 	       used in SVM	   	  	     	     	      |
|		 | 0/1 loss: l( f(x_i | \theta), y_i) = 1 <=> f(x_i | \theta) =/= y_i |
|		 |     	     used in theoretical analysis and definition of accuracy  |
+----------------+--------------------------------------------------------------------+
| Cost function: | More general, might be a sum of loss functions over a training set |
|      		 | plus sime model complexity penalty (regularization) 	 	      |
|		 | Mean Squared Error:						      |
|		 | 	MSE(\theta) = \frac{1}{M}\sum^N_{i=1}(f(x_i | \theta)-y_i)^2  |
|		 | SVM cost function: 				      		      |
|		 |      SVM(\theta) = ||\theta||^2 + C \sum^N_{i=1}\xi_i	      |
|		 | 	There are additional constraints connecting $\xi_i$ with $C$  |
|		 |	and with the training set.	 	    	    	      |
+----------------+--------------------------------------------------------------------+
| Objective      | The most general term for any function optimized during training.  |
| Function:	 | For example, a probability fo generating training set in maximum   |
| 		 | likelyhood approach is a well defined objective function, but it   |
|		 | is not a loss function nor cost function (however, you could       |
|		 | define an equivalent cost function).	    	      	  	      |
|		 | MLE is a type of objective function (which you maximize)	      |
|		 | Divergence between classes can be an objective function, but it is |
|		 | barley a cost function, unless you define something artificial,    |
|		 | like 1-Divergence, and name it a cost.    	       		      |
+----------------+--------------------------------------------------------------------+
| A loss function is a *part* of a cost function which is a *type* of objective	      |
| function.	       	      	   		       	    	      		      |
+-------------------------------------------------------------------------------------+


[Neural Network Tutorial][5] Notes
==================================

Activation function - The simulation of a neuron being turned on. So, it needs to be able
	   	      to change state from 0 to 1, -1 to 1 or from 0 to >0.
		      A common activation function is the sigmoid function, (however, it
		      is not much used in commercial applications).
		      \[
		      f(z) = \frac{1}{1 + \exp(-z)}

The networks are built in hierarchical structures with *nodes* (or neurons or perceptrons)
coupled with *weights*, and biases to each node. The output of such a node is the
activation function, where the input to it are the sum of all the weights multiplied with
the input. If we have input $\vec{x} \in \mathbb{R}^N$ the input to the activation function will be 
\[
z = b + \sum\limits^N_{i=1} w_i x_i
\]
The bias moves the activation, so we can control *when* it is activated. The weight (in
the 1-D example with sigmoid activation), changes the *slope* of the activation function.

Notation: 
    w_{ij}^{(l)} refers to the weight at the connection between node i in layer *l+1*, and
    node j in layer *l*. The order might seem odd, but the notation makes more sense when
    we add the bias.
    b_{i}^{(l)} is the bias weight to node i in the layer *l+1*.
    h_{j}^{(l)} is the output of node j in the layer *l*. (The output of the *activation
    function* $f(z)$.)
\[
h_{i}^{(l+1)} = f( b_{i}^{(l)} + \sum\limits^{N}_{j=1} w_{ij}^{(l)} x_{j})
\]





[1]: http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
[2]: http://adventuresinmachinelearning.com/tensorflow-eager-tutorial/
[3]: https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python
[4]: https://stats.stackexchange.com/questions/179026/
     objective-function-cost-function-loss-function-are-they-the-same-thing 
[5]: http://adventuresinmachinelearning.com/neural-networks-tutorial/


