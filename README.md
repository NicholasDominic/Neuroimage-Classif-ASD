# Transfer Learning using InceptionResNetV2 to the Augmented Neuroimage Data for Autism Spectrum Disorder Classification
General Information
- Contributors: **Nicholas Dominic, Daniel Daniel, Tjeng Wawan Cenggoro, Arif Budiarto, Bens Pardamean**
- Main Affiliation: [Bioinformatics & Data Science Research Center](https://research.binus.ac.id/bdsrc/) (BDSRC), Bina Nusantara University
- Programming Language: **Python 3.7.1**
- International Publication: [click here](http://scik.org/index.php/cmbn/article/view/5565)

## Frequently Ask Questions (FAQ)
- ***Can I copy all or some of your works?***
<br>>> You can, by specifying author's name as condition. Or simply do GitHub's fork from my repository.
- ***I can't open your .ipynb file, how to resolve the issue?***
<br>>> Copy the url (with .ipynb extension) and paste it to this [website](https://nbviewer.jupyter.org/).
- ***May I commit a change in your code?***
<br>>> You may. But only reliable changes will acquire my acceptance, as soon as possible.

## Contents

> ABSTRACT

From a psychiatric perspective, the detection of Autism Spectrum Disorders (ASD) can be seen from the differences in some parts of the brain. The availability of the four-dimensional resting-state Functional Magnetic Resonance Imaging (rs-fMRI) from Autism Brain Imaging Data Exchange I (ABIDE I) led us to reorganize it into two-dimensional data and extracted it further to create a pool of neuroimage dataset. This dataset was then augmented by shear transformation, brightness, and zoom adjustments. Resampling and normalization were also performed. Reflecting on prior studies, this classification accuracy of ASD using only 2D neuroimages should be improved. Hence, we proposed the use of transfer learning with the InceptionResNetV2 model on the augmented dataset. After freezing layer by layer, the best training, validation, and testing results were 70.22%, 57.75%, and 57.6%, respectively. We proved that the transfer learning approach was successfully outperformed the convolutional neural network (CNN) model from the previous study by up to 2.6%.

> RESULT CONCLUSIONS AND FUTURE WORKS

Our simple method to acquire and augment the only 172 neuroimages of ABIDE I dataset (NYU site) yielded an improvement from the previous method where the data used were the 1,992 neuroimages of ABIDE I and II. With smaller dataset fed to the denser model such as InceptionResNetV2 can achieve up to 78.9% of training accuracy, up to 58.9% of validation accuracy, and up to 57.56% of testing accuracy. The best configurations that can be reported were by leaving the parameters untrained and by changing the momentum value from .5 to .7. The improvement suggested that there is still ample room to prove that ASD can be detected from simpler brain scans (ours using the T2* or EPI), compared to the Functional Connectivity Matrix of associations between brain regions.

For future works, it is intriguing to expand the way of 2D/3D extraction from the origin 4D neuroimage. The development of a pre-trained model using the same base domain, i.e., brain scans/neuroimages instead of ImageNet can also be done since it has been proven to enhance performance and even lessen training time. Additionally, trying to construct a multimodal model where the patient phenotypic data are included may be worth further consideration.

> ACKNOWLEDGEMENT

All models were trained using NVIDIA Tesla P100 GPU (3584 CUDA cores) provided by [NVIDIA-BINUS Artificial Intelligence Research and Development Center](https://research.binus.ac.id/airdc/) (AIRDC).

## Author Detail Information
**NICHOLAS DOMINIC**
- **Education**: Graduate Student in **[BINUS University](https://mti.binus.ac.id/) Computer Science (AI stream)**
- **Email**: nicholas.dominic@binus.ac.id / dominicnick4@gmail.com / ndominic75@icloud.com
- **LinkedIn Profile**: [click here](https://www.linkedin.com/in/nicholas-dominic)

Do me a favor to share my works and freely contact me for further recognition. Have a great rest of your day!
