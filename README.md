# Autoencoder-with-GAN
Design and build a GAN, Combine Recurrent Autoencoder with Generative Adversarial Network.

# Introduction
In this project, I propose a novel model combining Recursive Autoencoder with GAN. I created two neural networks, the encoder and decoder of the autoencoder part, and one additional network that works as a binary classifier for the discriminator. In this way, the Autoencoder and the discriminator compete with each other in the game theory manner in order to improve the efficiency and accuracy of reconstructing input in the autoencoder. Furthermore, the output of the Autoencoder will be entered into the system as an input of the Autoencoder. My goal here is to fool the discriminator by a 50 percent chance for all recursive steps and update both the Autoencoder and discriminator’s parameters with respect to the competition between them.

# Methodology
## Autoencoder:  
Autoencoders first compress the data into a latent representation with lower dimensions. Then, it tries to reproduce the input data. Suppose we have a set of input data points ${x^1,x^2,…,x^m}$ each with many dimensions. The goal of the autoencoder is to map the input to some latent representation ${z^1,z^2,…,z^m}$ which has lower dimensionality than x, and also x can be reconstructed from it (we name the reconstructed data $\bar{x}$.) In order to talk about mapping more systematically, I propose z and $\bar{x}$ in the following way:<br />
```math
z^i= W_e  x^i+ b_e
```
```math
\bar{x}^i = W_d  z^i+ b_d
```
Where $W_e$ and $b_e$ are related to the encoder part, and $W_d$ and $b_d$ are for the decoder part of the autoencoder. <br />

We aim to reconstruct $\bar{x}^i$ to approximate $x^i$. So, the loss function is the sum of the squared difference between $\bar{x}^i$ and $x^i$. <br />
```math
$L(W_e,b_e,W_d,b_d) = \sum_{i=1}^m (\bar{x}^i - x^i)^2 = \sum_{i=1}^m ( W_d z^i + b_d- x^i)^2 = \sum_{i=1}^m ( W_d (W_e x^i+ b_e) + b_d - x^i)^2$ <br />
```
Therefore, minimizing this difference is the goal here, which can be done by stochastic gradient descent.

<img src="Images/FIG%201.png">

## The Discriminator: 
A Generative Adversarial Network is made of two neural networks – the generator and the discriminator - which compete with each other in the sense of game theory (Goodfellow et al., 2020). Presume training examples $x$ has an unknown distribution $p_input(x)$. The goal of the generator network is to learn $p_model(x)$ in a way that is as similar to $p_input(x)$ as much as possible. The output of the generator is defined by the generator function $G(u;W(G))$, where u is the input of the generator and $W(G)$ is a set of learnable parameters of the generator. In this project, the generator is the autoencoder's decoder part. <br />
The other part is the discriminator. The discriminator takes some examples $x$ as input and decides whether $x$ is real (drawn from the training samples) or fake (output of the generator). The discriminator function is $D(x;W(D))$ where $W(D)$ is a set of learnable parameters of the discriminator. <br />

Training of GANs consists of evaluating both parameters of the discriminator, which maximizes its accuracy, and also the parameters of the generator that can maximally fool the discriminator (Creswell et al., 2018). <br />
The cost of training is defined by a cost function $V(G,D)$ that depends on both the generator and the discriminator. Here, I decided to use minimax GAN, which is based on the minimax game. This type of training solves the bellow equation: <br />
```math
\left( \max_D \left(⁡min_G (V(G,D) \right) \right)
```
where $V(G,D) = E[P_{input}(x)] log⁡ D(x) + E[P_{model}(x)] log⁡(1-D(x))$. <br />


Here $E[P_{input}(x)]$ is the expectation over the input distribution, $E[P_{model}(x)]$ is the expectation over the autoencoder’s output distribution, and $log(D(x))$ is the log-likelihood of the discriminator. In the training process, the parameters of one network are frozen while the other network’s parameters are being updated (Creswell et al., 2018). <br />

Goodfellow et al.4 showed that for a generator with probability distribution $P_{model}$ there is a unique optimal discriminator, $D'(x) = \dfrac{P_{input}(x)}{P_{input}(x) + P_{model}(x)}$. They also showed that when $P_{input}(x)$ is equal to $P_{model}(x)$, the generator is optimal, which then leads them to the optimal discriminator with $D^*(x) = 0.5$. In other words, the generator is optimal when the discriminator is totally confused between the real samples and the fake ones, which, in fact, makes the discriminator make a 50-50 chance decision.

<img src="Images/FIG%202.png">

## Combining Autoencoder with Discriminator: 
In this project, I combined autoencoder with GAN in the way that we have the normal autoencoder with a discriminator, which defines the accuracy of our resampling (i.e., recreating the input via autoencoder). Unlike the traditional architecture in GAN, in which the generator competes with the discriminator, the whole autoencoder competes with the discriminator in this case. In this matter, after training networks individually, one of the two networks – The autoencoder or the discriminator – is frozen, and the other network’s parameters are updated. So, if the discriminator can recognize fake data with more than 50% probability, the discriminator’s parameters will be fixed, and in order to improve the autoencoder, only the autoencoder parameters will change. <br />

As an extension to Autoencoders, researchers introduced a “Recurrent autoencoder” (Susik, 2021). In the Recurrent Autoencoders, the generated output of the decoder will be the input of the autoencoder, and the final output of the autoencoder after the k recursive process will be x'^k. In regard to this change, the loss function will be changed to $L(W_e,b_e,W_d,b_d) =(x'^k - x^2)$ instead of $(x'- x)^2$. The goal here is to minimize the loss function by updating the encoder and decoder’s parameters. <br />

In this project, I proposed a novel model combining Recurrent Autoencoder with GAN in the way they both compete with each other in order to improve the efficiency and accuracy of reconstructing input in autoencoders. In this matter, all the outputs from the decoder (from the first output, $x'^1$ to the last recursive output, $x'^k$, will be the input of the discriminator to decide whether it is fake or real (the figure below). My goal here is to fool the discriminator by a 50 percent chance for all recursive steps and update both the Autoencoder and discriminator’s parameters in respect of a competition between the Recursive Autoencoder and the discriminator. <br />

<img src="Images/FIG%203.png">

# Implementation

## Dataset: 
For implementing this project, I used Python language and Pytorch library for creating my networks. For the first step, I decided to use the MNIST dataset to create a reliable autoencoder. MNIST dataset contains 28*28 pixels images, which are greyscale. For the second step, I used the CIFAR10 dataset to have colorful RGB images to find out about the performance of the autoencoder when working with RGB images. CIFAR10 dataset has 32*32 pixels images, which have 3 channels, each of them for blue, green, and red, respectively. Below, I will dive into the details of implemented networks and the results they generated. <br />
It is worth to mention that for the autoencoder of all parts I used Adam optimizer with learning rate equals to 0.005. I also used Adam optimizer for the discriminator with the learning rate of 0.001.

## Autoencoder
In the first part, I implemented an autoencoder with the MNIST dataset. The encoder of this part consists of 3 convolutional layers and 3 max pool layers in between them, which produce 3*3 images with 64 channels at the end, and also 3 fully connected layers, which produce 4*4 images. The decoder consists of 3 fully connected layers, which finally produce 3*3 images with 64 channels, and then 3 convolutional layers convert them to 28*28 images. <br />
In the next part of the project, I decided to change my data set to a more complex dataset than MNIST. Therefore, I changed my dataset to CIFAR10.  The autoencoder of this part consists of an encoder and a decoder, each of them having 7 convolutional layers. I used a 3*3 filter with padding equal to 1 and stride equal to 1 in order to maintain the dimensions of the data. Also, a 4*4 filter with padding equal to 2 and stride equal to 2 was used to half the width and height of the images. Therefore, for the encoder, after the first layer, the data has 8 channels, and after each 2 layers, the channels of the data will be doubled as the dimension will decrease to 1⁄4 its size. For the discriminator, layers act as the reverse of the encoder layers. Finally, the image's dimensions will be decreased to 2⁄3 its size. 

## Recursive Autoencoder
In this part, I used the previous autoencoder to feed its output to the network again as the input and build the recursive autoencoder. I implemented a recursive autoencoder with 2 recursive steps. For the training, in each epoch, when iterating through the data, for each data I compute the loss with different recursive steps and then change the weights of the network for optimization. Also, I start this way of training randomly with 0, 1, or 2 steps. In this way, assume that it starts training with 0 recursive steps; for the first image, it computes loss without any recursive steps; for the second image with 1 recursive step; for the third image with 3 steps, and this cycle will start again for the rest of the images. As a result of this training, the loss function will be the sum of losses with 0, 1, and 2 steps with the same weights. 

## Autoencoder with GAN
In this part, I first tried the MNIST dataset. For the GAN part, I built a discriminator, which was a binary classifier and consisted of four fully connected layers that convert 28*28 images into just one node in order to classify the images as fake or real. I implemented GAN in a way that the discriminator competes with the whole autoencoder, not just the decoder part. Training these networks is based on game theory rather than optimization. The autoencoder of this part was the same autoencoder used before for the MNIST dataset.
I also used the autoencoder with GAN for the CIFAR10 dataset. Here, the discriminator is just like the discriminator for the MNIST data set, and the autoencoder is the same as the autoencoder used before for the CIFAR10 dataset. The way of training this network is a little different than that of using GANs. First, I train the autoencoder in order to create good reconstructed images. Then, I trained the discriminator with real data, which is the CIFAR10 dataset, and fake data, which is data that had been reconstructed with the use of an autoencoder. After that, the autoencoder will be trained again in order to produce the output that fools the discriminator. Therefore, the first part is unsupervised learning for autoencoders, and the second part is supervised learning for both discriminators and autoencoders.

# Results

## Autoencoder
For the MNIST dataset, I used Mean Squared Error for the loss function. The loss of the autoencoder after training it with 3000 images and 100 epochs and testing with 600 images was 0.014. The reconstructed images had great shapes and luminosity (because the images are greyscale in the MNIST dataset). However, I needed to change the network architecture in order to improve the performance of the CIFAR dataset. The diagram of the training loss validation loss and reconstructed images for the last two epochs can be found in the figures below, respectively. <br />

<img src="Images/MNIST%20-%20AE%20-%202.png">
<img src="Images/AE%20-%20attempt4%20-%20pic1.png">

For the CIFAR10 dataset, I used Binary Cross Entropy for the loss function. The loss of the autoencoder after training it with 10000 images and 200 epochs and testing with 2000 images was 0.553. The reconstructed images had a little shape, and the color faded a little bit as a result of the compressive nature of the autoencoder. The reconstructed images for the last epoch and diagram of the training loss and validation loss can be found in the figures below, respectively.

<img src="Images/MNIST%20-%20AE%20-%201.png">
<img src="Images/AE%20-%20attempt4%20-%20pic2.png">

## Recursive Autoencoder
For the recursive part, I wanted to observe how this way of training the autoencoder will affect its performance it. I used BCE for the output of the autoencoder without any recursive steps. After training the recursive autoencoder with 10000 images and 150 epochs, the loss was 0.559. The reconstructed images have better shapes than the autoencoder without any recursive steps, but the color of the images faded a little bit. The diagram of the training loss validation loss and reconstructed images for the last epoch can be found in the figures below, respectively.

<img src="Images/RecurrentAE%20-%203%20steps%20-%20attempt3%20-%20pic2.png">
<img src="Images/RecurrentAE%20-%203%20steps%20-%20attempt3%20-%20pic1.png">

## Autoencoder with GAN
For the last part, I implemented it with both the MNIST dataset and the CIFAR10 dataset. For the MNIST dataset, I used BCE for the loss function of the discriminator and MSE for the autoencoder. The loss of the autoencoder after training it with 7000 images and 200 epochs and validating it with 2000 images was 0.509. The reconstructed images got a little blurry, and the loss was higher than the regular autoencoder with the MNIST dataset. It may be because of the perfect discriminator I had for the MNIST dataset. The discriminator loss here was 0.02, and the autoencoder could not fool the discriminator. This may be because the MNIST dataset images are not complex. The reconstructed images for the last epoch can be found in the figure below. <br />

<img src="Images/MNIST%20-%20AE%20and%20GAN%20-%201.png">

When using the CIFAR dataset, I used BCE for loss, too. In each epoch, I first train the autoencoder just like before, then train the discriminator and autoencoder to make them compete with each other. After training, which was over with 20000 images and 20 epochs, I tested just the autoencoder to evaluate its performance. The loss of the autoencoder was 0.567. The reconstructed images lost color in some cases compared to the autoencoder without the discriminator, and the images became blurrier. The reconstructed images for the last epoch and diagram of the training loss and validation loss can be found in Figure 11 and Figure 12, respectively.

<img src="Images/RecurrentAE%20-%203%20steps%20-%20attempt3%20-%20pic1.png">
<img src="Images/RecurrentAE%20%20with%20GAN-%202%20steps%20-%20attempt1%20-%20pic2.png">

## Comparing three networks
In conclusion, the results of the three networks should be compared to each other. As mentioned above, losses for the three networks are close to each other, and no major differences could be found. The reconstruction of images for all three networks are look the same, however, autoencoder without GAN and recursive steps looks like to perform a little better than the other with reconstruction of shapes. Autoencoder with recursive steps looks like it generates colors better than the others, but the shapes got blurrier. Finally, the combination of autoencoder with GAN does not outperform any of the other networks. It could be because the recursive autoencoder and autoencoder with GAN need more data and epochs to learn the reconstruction better, but I used Google Colab and had a limited amount of RAM and computational power. The tables comparing three networks for the CIFAR dataset can be found in the table below.

| Neural Network Type | Average Loss (BCE) |
| --- | --- |
| Autoencoder | 0.553 |
| Recursive Autoencoder	| 0.559 |
| Autoencoder with GAN | 0.593 |

# Conclusion
In this project, I combined autoencoder with GAN in the same way that we have the normal autoencoder with a discriminator, which defines the accuracy of our resampling. Unlike the traditional architecture in GAN, in which the generator competes with the discriminator, here, the whole autoencoder competes with the discriminator in the game theory manner. First, I implemented autoencoder, then recursive autoencoder with the use of that particular autoencoder. After those two steps, I built a discriminator and made it compete with the autoencoder. For the first attempt, I used the MNIST dataset and then switched to the CIFAR dataset to observe the network’s performance for RGB images. As a result of the above-mentioned implementation, we can see that images have better shape when reconstructing images with recursive autoencoder, but their color becomes faded a little bit. I believe that by working in a better programming environment and having more RAM and better-performance GPUs, deeper and more complex networks can be used to have better performance. As we can see, the recursive autoencoder reconstructs shapes better, so there is a high hope of better performance with the recursive autoencoder.

# Limitation and Future Works
I used Google Colab to run my codes. One of the important limitations was the limited amount of RAM that Google Colab allows you to use. Because of that, I could not train my networks with larger datasets or more epochs. Perhaps using a better device that has more RAM and more GPU cores will improve the accuracy of the networks. With a better environment for running codes, deeper and more complex architecture can be used, as well as larger datasets. In future works, I might use larger, more complex image datasets like ImageNet. Having a deeper network with more convolutional layers is another work that could be done to help improve the autoencoder. Also, it is worth mentioning that fully connected layers could be added to this network to see the results of adding fully connected layers to the CNN that has been built before.

# References
Radford, A., Metz, L., Chintala, S. (2016). Unsupervised Respresentation Learning with Deep Convolutional Generative Adversarial Network, http://arxiv.org/abs/1511.06434 <br />
Bourlard, H., & Kamp, Y. (1988). Auto-association by multilayer perceptrons and Singular Value Decomposition. Biological Cybernetics, 59(4-5), 291–294. https://doi.org/10.1007/bf00332918 <br />
Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., & Krishnan, D. (2017). Unsupervised pixel-level domain adaptation with generative Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2017.18 <br />
Creswell, A., White, T., Dumoulin, V., Arulkumaran, K., Sengupta, B., & Bharath, A. A. (2018). Generative Adversarial Networks: An overview. IEEE Signal Processing Magazine, 35(1), 53–65. https://doi.org/10.1109/msp.2017.2765202 <br />
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2020). Generative Adversarial Networks. Communications of the ACM, 63(11), 139–144. https://doi.org/10.1145/3422622 <br />
Kramer, M. A. (1991). Nonlinear principal component analysis using auto-associative Neural Networks. AIChE Journal, 37(2), 233–243. https://doi.org/10.1002/aic.690370209 <br />
Lopez Pinaya, W. H., Vieira, S., Garcia-Dias, R., & Mechelli, A. (2020). Autoencoders. Machine Learning, 193–208. https://doi.org/10.1016/b978-0-12-815739-8.00011-0 <br />
Susik, R. (2021). Recurrent autoencoder with sequence-aware encoding. Computational Science – ICCS 2021, 47–57. https://doi.org/10.1007/978-3-030-77964-1_4 <br />
Tang, T.-W., Kuo, W.-H., Lan, J.-H., Ding, C.-F., Hsu, H., & Young, H.-T. (2020). Anomaly detection neural network with dual auto-encoders gan and its industrial inspection applications. Sensors, 20(12), 3336. https://doi.org/10.3390/s20123336 <br />
Zhu, J.-Y., Krähenbühl, P., Shechtman, E., & Efros, A. A. (2016). Generative visual manipulation on the natural image manifold. Computer Vision – ECCV 2016, 597–613. https://doi.org/10.1007/978-3-319-46454-1_36
