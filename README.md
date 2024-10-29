# 3DImageReconstruction
#Contributed by Shubh Patel and Tejas Patel.

#We are working on a deep learning project where we are going to reconstruct the object in 3-d 
#using the 2-d images using deep learning.  The goal for this project is to take 2D images which 
#are taken from different angles as input and reconstruct their 3D shapes of the images.  
#We think that this problem is an important task to work on because of the following reasons-   
#1. first of all, Most of the visual data which has been captured by the devices in our daily life are 
#of the 2D, while on the other hand object exist in the 3D, So bridging the gap between two can 
#help to create a realistic model in AR/VR, Gaming, Robotics, and many more such field.  
#2.Another place where this can be useful is in 3D printing. Instead of creating a blender 3D 
#model, one can input the 2D image which will generate 3D printable Model for 2D Image. 
#For this project we are planning to use Pix3d dataset, which has 9 different categories such as 
#bed, bookcase, chair, and many more.  The dataset includes 2D images along with its 3D model 
#of the form “.obj”.   
#The Methods or Algorithms which we plan to include are 3D CNN, Point Net, Point Net++, Data 
#Augmentation Techniques (For enhancing The Structure Of 2D Image) such as Random 
#Rotation. Translations, Scaling, Color, Jittering, Voxel Representation (It Will be useful for 
#converting 3D Models into voxel grids). 3D CNNS and Point Net Are the Architectures which we 
#will be using mainly for the problem of 3D Object Reconstruction. We will use other techniques 
#such as Intersection Over Union (IOU), Chamfer Distances, etc. 
#For the Evaluation process of our project/research we are going to use evaluation metrics such 
#as Chamfer Distance (CD), Intersection Over Union (IOU), Voxel Accuracy, F-1 Score, K -Fold 
#Validation. Then We think that the physical evaluation is also quite important for this project, 
#manually inspecting the generated 3D objects.
