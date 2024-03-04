# A-Machine-Learning-Approach-to-Investigate-the-Evolution-of-Sunspots
This repository contains all the relevant code for my MSc.
The contents of the repository are the following:
- **Transformer folder:** Contains the Transformer Encoder with all modules needed.
- **DataSet.py:** Data set class used to train the hybrid model.
- **Hybrid.py:** Definition of hybrid model containing the vgg16 and transformer encoder, used for sequence classification.
- **GradCam.py:** Implementation of Grad-CAM, Guided Backpropagation and Guided Grad-CAM.
- **specs.txt:** File containing the parameters used for training.
- **Meta_Labeling.py:** Algorithm applied to get sequence-wise labels from the image-wise labeled observations.
- **reduce_dataset.py:** Algorithm applied to equalize amount of positive and negative sequences.
- **Hybrid_training.py:** Algorithm to train the hybrid model.
- **Sunspot_detection_model:** Model used to perform image wise classification.
- **adaptive_meta_labeling:** Class to perform sequence-wise classification


Unfortunately GitHub does not allow to upload the trained weights for the hybrid model, since the file is too large.
