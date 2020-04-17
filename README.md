# MRI Motion Correction
### MSAI Practicum project with NU Radiology
In collaboration with Radiology department at Northwestern Medicine Feinberg Medical School, this project's goal is to develop a computational model that can correct for blurring as a result of motion introduced to MRIs during their capture. Given the time it takes to complete an MRI scan, any movement by the patient can disturb the resulting image, diminishing the value it provides the physicians. By designing a model that can correct for this motion, it can reduce the number of subsequent MRIs the patients have to endure (reducing radiation exposure) in order to get a clear image, as well as providing a method of getting a clear image from patients who are unable to stay still.

## Part 1- Data Generation
In order to train a model for this purpose, a dataset of ground-truth clear mris and their corresponding blurred mris is near impossible. It is unreasonable to subject people to unnecessary radiation exposure to gather either a purposely blurred image with a corresponding clear image, or clear mri from a patient who is unable to stay still. To this end, I developed a computational method of generating blurred mris using clear mris and motion files recording actual human movement. This method incorporates movement that was recorded from actual patients, onto the k-space of the mri, simulating what happens during actual mri capture. Here is an example of a 2d slice of a clear mri and the blurred image that is generated from this method:
![image](https://user-images.githubusercontent.com/30561629/79600123-4515ac80-80ac-11ea-94b2-2c64f550043b.png)
On the left is the clear 2d slice, and the right is the generated blurred image.

With this method, we can not only generate realistic blurred mris, but we can augment the amount of training data by using multiple motion files on each of the clear mris as well. This provides a method to augment the data, minimizing the number of novel mris required for this project.

## Part 2- Image Reconstruction
Now that the training data is created, the model used to correct for the motion was created using the U-Net architecture, trained with Perceptual Loss.
### U-Net
![image](https://user-images.githubusercontent.com/30561629/79601142-f537e500-80ad-11ea-8c60-6d2d35c95824.png)

This architecture was selected because U-Net has proven to do well with segmentation in medical images. The architecture is also a good fit given the input and output shapes are the same, which is what is needed for this project. And by using a linear activation on the last convolution layer instead of the sigmoid activation which is used for classification, the model can provide continuous outputs for each pixel.

### Perceptual Loss
![image](https://user-images.githubusercontent.com/30561629/79601550-97f06380-80ae-11ea-8e85-358f023b206f.png)

U-Net generally uses cross-entropy loss, which works well for image segmentation (classification), but is not appropriate for our goal. MSE would be more appropriate, but is insufficient in requiring the network to produce clear images, which is our main goal. Perceptual Loss improves upon this by leveraging the features extracted from the VGG19 network, which is known to be able to pull out features such as edges in the earlier layer activations, to full shapes in latter layer activations. By putting the U-Net output through VGG as well as the target clear image through VGG, extracting the activations from several layers, and calculating MSE between the features, the results are great.

Training on 2500 2d slices over 80 epochs, the model reaches a **MSE of 3.2920e-04**.

Here are some example outputs from the trained model on a held-out validation mri:
![image](https://user-images.githubusercontent.com/30561629/79598769-1e567680-80aa-11ea-8116-f5fd1a756c7e.png)
![image](https://user-images.githubusercontent.com/30561629/79600699-3380d480-80ad-11ea-979d-d418ffe3f944.png)
![image](https://user-images.githubusercontent.com/30561629/79600931-9e321000-80ad-11ea-88f3-582d08bb0842.png)
On the left is a clear 2d slice, the middle is blurred image generated from part 1, the right is the motion-corrected image output from the model.

As you can see, the model does very well cleaning up the white-matter and removing the artifacts outside of the skull.
