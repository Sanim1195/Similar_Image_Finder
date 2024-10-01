# Finding Similar Images

    We encode all images into vector space and then find high density regions in this vector space, i.e., regions where the images are fairly similar.

# Steps:

1. To compare images, we compute dense representations (embeddings) for them. These embeddings capture   the essential features of an image and map it to a vector in a high-dimensional space. I am using the clip module. The clip module has an callable object i.e preprocess : A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input 


2. Once we have these embeddings, we can use a similarity metric to compare them. A popular choice is the cosine similarity.Cosine similarity measures the angle between two vectors. If the angle is small (close to 0°), the vectors are similar; if it’s large (close to 180°), they are dissimilar

    