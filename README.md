# Finding Similar Images

    We encode all images into vector space and then find high density regions in this vector space, i.e., regions where the images are fairly similar.

# Steps:

1. To compare images, we compute dense representations (embeddings) for them. These embeddings capture   the essential features of an image and map it to a vector in a high-dimensional space. I am using the clip module. The clip module has an callable object i.e preprocess : A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input (Yeah Right a model with it's own pre processing and tensor conversion). Download the clip module from here: https://github.com/openai/CLIP?tab=readme-ov-file



2. Once we have these embeddings, we can use a similarity metric to compare them. A popular choice is the cosine similarity.Cosine similarity measures the angle between two vectors. If the angle is small (close to 0°), the vectors are similar; if it’s large (close to 180°), they are dissimilar. I have also included Euclidean distance to find close copies of an image but might not be the original as screenshots of original images could have slightly different pixels and noise.


3. I don't really understand why i might need clustering but to quench my thirst i tried KMeans clustering which was alright but may be i can try grouping based on simialrity. And looking into other clustering algos out there.



# RESOURCE:
    https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
    https://www.sbert.net/docs/sentence_transformer/loss_overview.html
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/image-search/Image_Duplicates.ipynb
    https://www.learnpytorch.io/
    https://pytorch.org/docs/stable/nn.functional.html


# Similarity Webdocs:
    https://en.wikipedia.org/wiki/Content-based_image_retrieval
    https://en.wikipedia.org/wiki/Color_histogram


# Matplot:
    https://matplotlib.org/stable/plot_types/basic/scatter_plot.html#sphx-glr-plot-types-basic-scatter-plot-py
    
    