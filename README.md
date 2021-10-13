# DCGAN-on-Celeba-dataset
#### Here I applied Deep Convolutional Generative Adversarial Networks (DCGANs) on the famous Celeba dataset using Pytorch. The reference and model for my project was taken from the paper, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz and Soumith Chintala. 
Link for the paper - https://arxiv.org/abs/1511.06434

Link for the data - https://www.kaggle.com/jessicali9530/celeba-dataset

## Architecture of DCGAN

![1 5ALjnfAqwcWbOsledTBXsw](https://user-images.githubusercontent.com/27720480/137178394-2db779f7-919e-4927-a249-7ee4cba07a25.png)
#### The above pic is of the architecture of Generator. Discrimiator has the same layout but just opposite flow. The job of Generator is to take in random noise and turn it to something like real data while the job of Discriminator is to taken in data and give the probability of it being real. 

## Training 
#### For training our DCGAN, we took in 64*64*3 real images from Celeba dataset and 64*64*3 fake images from the generator and fed them to the discriminator. We used BCELoss or Binary Cross Entropy Loss as it's very similar to the loss mentioned in the paper and works just like it after doing some mods to the target. We used Adam as our optimizer for training with a learning rate of 2e-4. We also did some modification to the hyperparameters in accordance to the paper. 

#### I trained for a total of 10 epocs on 60,000 images. Due to high computational cost, I was not able to train on all the images of Celeba dataset. The training took me around an hour and gave me okay resuls which are shown below. 

## Results
<img width="391" alt="Screenshot 2021-10-13 223746" src="https://user-images.githubusercontent.com/27720480/137180512-f7871f8a-6fa6-4fbe-83df-219ae68da687.png">
#### As you can see above, I got okay results which can be improved by increasing the number of epochs and also by increasing the data size. 
#### One thing to note is that GANs are highly sensitive to the hyperparameters. 
