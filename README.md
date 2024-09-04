# comp-vis
To run the code: python runHW6.py all

Based on a Computer Vision based course I took, this code takes 2 images of an object and will return a 3d point cloud. 

This is done by estimating the relative camera position in the 2 images of the object where the camera pose will be used to triangulate teh depth of each point in the scene. 

After computing the fundamental matrix between the 2 images, using the intrinsics, we can compute the essential matrix between the images. The essential matrix can then be decomposed to be split into a rotation and translation from one image to the other. 

We are using SIFT in order to identify point correspondences in both images, and RANSAC to estimate the fundamental matrix, as SIFT contains some errors in point correspondences

Now that we have the relative pose between the 2 images, we need to find a large population of point correspondences between the images. One would think that we would have to search the entire image in order to find the original point, but since a point in the left image corresponds to a line in the right image, we only have to search across that epiline. So rather than going pixel by pixel, we can template match by using a sliding window along the epiline and computing the normalized cross correlation at each location. We know there is a match when the cross correlation is at its max. 
