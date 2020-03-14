# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/explore1.png "Training images"
[image2]: ./images/explore2.png "Histogram"
[image3]: ./images/preprocess.png "Preprocessed"
[image4]: ./images/test_images.png "Test Images"
[image5]: ./test_images/test01.jpg "Test Image 1"
[image6]: ./test_images/test02.jpg "Test Image 2"
[image7]: ./test_images/test03.jpg "Test Image 3"
[image8]: ./test_images/test04.jpg "Test Image 4"
[image9]: ./test_images/test05.jpg "Test Image 5"

## Rubric Points
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is `34799`.
* The size of the validation set is `4410`.
* The size of test set is `12639`.
* The shape of a traffic sign image is `(32, 32, 3)`.
* The number of unique classes/labels in the data set is `43`.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I randomly chose 20 images among training dataset and I added the corresponding labels on the their titles.

![alt text][image1]

I wanted to inspect three datasets' distribution, so I obtained the following figures.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I converted a RGB image to a grayscale image using `skimage.color.rgb2gray`, since human can recognize traffic signs in grayscaled image. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

In addition, I decided to do one-hot encoding by `sklearn.preprocessing.LabelBinarizer`, because The difference between the original label data set and the one-hot encoded data set is the following.

```python
print(traffic_signs[y_train[2333]])
print(y_train[2333])
Y_train = lb.transform(y_train)
print(Y_train[2333])
```

If I run the above code, I get

```
Speed limit (30km/h)
1
[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0]
```
You can notice a vector-valued data instead of an integer data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (this is a modified LeNet5, because I included one convolution layer to the LeNet5):

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32 grayscale image   						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 11x11x16 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 7x7x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x32  					|
| Flatten				|      		   outputs 288 						|
| Fully connected		| 120 nodes    outputs 120 						|
| Fully connected		| 84  nodes    outputs 84						|
| Fully connected		| 43  nodes    outputs 43						|
| Softmax				|              outputs 43        				|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the mean of cross-entropy as a loss function. I applied stochastic gradient descent (SGD) with batch size 32 and 32 epochs. `AdamOptimizer` was used to minimize the mean of cross-entropy loss function with learning rate 0.001. You can verify easily by the following code snippet.


```python
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, fc3))
optim = tf.train.AdamOptimizer(learning_rate=LR)
train = optim.minimize(loss)
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Here is my record for the training.
```
Epoch 0: (train: 0.930, valid: 0.860)
Epoch 1: (train: 0.969, valid: 0.919)
Epoch 2: (train: 0.983, valid: 0.936)
Epoch 3: (train: 0.981, valid: 0.924)
Epoch 4: (train: 0.986, valid: 0.941)
Epoch 5: (train: 0.990, valid: 0.925)
Epoch 6: (train: 0.992, valid: 0.934)
Epoch 7: (train: 0.991, valid: 0.941)
Epoch 8: (train: 0.989, valid: 0.945)
Epoch 9: (train: 0.993, valid: 0.945)
Epoch 10: (train: 0.994, valid: 0.947)
Epoch 11: (train: 0.998, valid: 0.960)
Epoch 12: (train: 0.991, valid: 0.954)
Epoch 13: (train: 0.998, valid: 0.953)
Epoch 14: (train: 0.996, valid: 0.951)
Epoch 15: (train: 0.998, valid: 0.958)
Epoch 16: (train: 0.995, valid: 0.944)
Epoch 17: (train: 0.991, valid: 0.949)
Epoch 18: (train: 0.998, valid: 0.953)
Epoch 19: (train: 0.995, valid: 0.951)
```
I choose epoch 11 because it has the highest validation accuracy.

My final model results were:

* training set accuracy of 99.8%.
* validation set accuracy of 96.0%.
* test set accuracy of 92.7%.

If a well known architecture was chosen:

* What architecture was chosen?
> LeNet5 + modification
* Why did you believe it would be relevant to the traffic sign application?
> I think the recognition of an image is pretty common sense, so I chose LeNet5 for this problem. However, it did not give enough accuracy for this udacity project. That is why I added an additional convolution layer into LeNet5.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
> I think my result is not over-fitted because accuracies of train and validation datasets. Their accuracies are over 93%, and test accuracy is 93.8%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:

![alt text][image4]

I chose `test04.jpg` because of its ambiguity, in other words, it is hard to classify what number it is even for me. It might be 30 km/h sign. The rest of images are expected as easy test images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I chose five sings for this question (see the attached images).

![alt text][image5] ![alt text][image6]![alt text][image7] ![alt text][image8] ![alt text][image9]


Here are the results of the prediction:

| Image			        		 			|     Prediction	        					|
|:-----------------------------------------:|:---------------------------------------------:|
| General Caution (`test01.jpg`)      		| General Caution   							|
| Slippery Road (`test02.jpg`)				| U-Slippery Road 								|
| 100 km/h (`test03.jpg`)					| 100 km/h										|
| 30 km/h (`test04.jpg`)					| 50 km/h					 					|
| Stop (`test05.jpg`)	 					| Stop      									|


The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of about 92%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the following code block.
```python
prediction = tf.argmax(fc3, 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './checkpoints/traffic_classifier_{}.ckpt'.format(epoch_selection))
    for i, test_image in enumerate(test_images):
        img_path = './test_images/' + test_image
        img = mpimg.imread(img_path)
        resize_img = resize(img, (32, 32))
        gray_img = rgb2gray(resize_img).reshape(-1, 32, 32)
        pred = sess.run(prediction, feed_dict={x: gray_img})
```

For the first image, the model is definitely sure that this is a general caution (probability of 1.0). The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| General Caution   							|
| 0.00     				| Right-of-way at the next intersection			|
| 0.00					| Pedestrians									|
| 0.00	      			| Traffic signals				 				|
| 0.00				    | Roundabout mandatory  						|


The second image is also determined that this is a slippery road by the model. The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Slippery road 	  							|
| 0.00     				| Bicycles crossing								|
| 0.00					| Dangerous curve to the right					|
| 0.00	      			| Road work						 				|
| 0.00				    | Dangerous curve to the left					|

For the third image, the model is sure of a sign of 100 km/h with probability 0.86. A sign of 80 km/h could be used with probability 0.14. The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.86        			| 100 km/h 			  							|
| 0.14     				| 80 km/h 										|
| 0.00					| No passing for vehicles over 3.5 metric tons	|
| 0.00	      			| Ahead only					 				|
| 0.00				    | 120 km/h										|


The fourth test image is interesting since it is confusing between 50 km/h and 30 km/h. Even I also am confused to determine this sign with 100%. The model is saying that this sign might be 50 km/h with probability 0.65 (quite low). The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.65        			| 50 km/h 			  							|
| 0.35     				| 30 km/h 										|
| 0.00					| 80 km/h 										|
| 0.00	      			| 100 km/h						 				|
| 0.00				    | 120 km/h										|

The last image is easily determined as a stop sign by the trained model. The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Stop  			  							|
| 0.00     				| No entry 										|
| 0.00					| No vehicles									|
| 0.00	      			| 60 km/h						 				|
| 0.00				    | Yield 										|

