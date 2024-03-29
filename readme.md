## Disk classification

**Task**: Given an input image, output the type of disk in it, and the location of the disk.

Non-commercial use only!

## Requirement 
pytorch==1.13



## Train and test
1. Crop the image using DataProcessing.py. All images will be saved in Crop folder.
    python DataProcess.py
2. Train the model
    python train.py
3. Test the model on new images
    python test.py Disks\2\image1.jpg

