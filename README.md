# Finding Similar Images

    We encode all images into vector space and then find high density regions in this vector space, i.e., regions where the images are fairly similar.

# Steps:
    1. To compare images, we compute dense representations (embeddings) for them. These embeddings capture   the essential features of an image and map it to a vector in a high-dimensional space.I am using the imgbeddings library (https://github.com/minimaxir/imgbeddings?tab=readme-ov-file) 
    
    -> By default, imgbeddings will load a 88MB model based on the patch32 variant of CLIP, which separates each image into 49 32x32 patches.Also it does require the image to be ina  square shape and if not the biasesness can be very high. 

    -> So i am using PIL library now

    2.Once we have these embeddings, we can use a similarity metric to compare them. A popular choice is the cosine similarity.Cosine similarity measures the angle between two vectors. If the angle is small (close to 0°), the vectors are similar; if it’s large (close to 180°), they are dissimilar

                                    OR
    3.Vision Models and Image Encoders:
    To compute these embeddings, we use vision models (also known as image encoders). These models learn to represent images in vector space. It can also encode both images and text into the same vector space, enabling tasks like image-text matching.