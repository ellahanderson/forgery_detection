My project attempted to build a convolutional neural network to detect a real Camille Corot painting from a forgery of a Camille Corot painting.

This repository contains all of the code used for my project. The images from the dataset are too large to upload to GitHub, so they are linked in a Google Drive. 

Paintings: Linked here: https://drive.google.com/drive/folders/1giR--W5x0i7d2jKzh1peH3-a6AyzJAcR?usp=share_link

The "paintings" folder contains the dataset I built. It is split into test, train, and validation sets using an approximate 70-20-10 split. Each of these categories has two subfolders - one for Corot paintings and one for forgeries.

The images were segmented for use in the brushstroke model using the files "brushstrokes_corot_test.py", "brushstrokes_corot_train.py", "brushstrokes_corot_validation.py", "brushstrokes_forgery_test.py", "brushstrokes_forgery_train.py", and "brushstrokes_forgery_validation.py". There are thousands of segments created from this, which is too many to upload to Google Drive without running out of space, but the segments can be replicated by running the code in these python folders with the dataset linked in Google Drive.

I tried each version of the model - using paintings as a whole and using brushstrokes - using RGBA, grayscale, and RGB color spaces to see which had the highest accuracy. After running those models, I also printed out the classification report.

The "corot.py" file is the original model I used. That has the default color space, RGB. 

The "in_grayscale.py" file is the model from corot.py edited to be in the grayscale color space.

The "rgba_model.py" file is the model from corot.py edited to be in the RGBA color space.

Each of these also has a linked confusion.py file to plot the confusion matrix and print the classification report.

"brushstrokes_model.py" contains the model using the segmented paintings to focus on brushstoke features instead of focusing on shape features like in the models with the entire paintings. Since there are so many segments, it was possible to edit the model since it could handle more epochs and steps. I tried each of the three color spaces for brushstrokes and chose to use RGBA since it had the highest accuracy.

There are also output images of graphs for each model, named accordingly.




