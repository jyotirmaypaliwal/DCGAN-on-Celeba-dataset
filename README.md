# DCGAN-on-Celeba-dataset
#### Here I applied Deep Convolutional Generative Adversarial Networks (DCGANs) on the famous Celeba dataset using Pytorch. The reference and model for my project was taken from the paper, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz and Soumith Chintala. 
Link for the paper - https://arxiv.org/abs/1511.06434

## Architecture of DCGAN

![1 5ALjnfAqwcWbOsledTBXsw](https://user-images.githubusercontent.com/27720480/137178394-2db779f7-919e-4927-a249-7ee4cba07a25.png)
#### The above pic is of the architecture of Generator. Discrimiator has the same layout but just opposite flow. The job of Generator is to take in random noise and turn it to something like real data while the job of Discriminator is to taken in data and give the probability of it being real. 
