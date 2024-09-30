from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

def calculat_distance(image1, image2):
    """ Calculates Euclidean Distance between 2 images """
    return np.sum((image1-image2)**2)

def calculate_cosine_similarity(image1, image2):
    

original_image = "assets/photos/original.jpg"
copied_image = "assets/photos/screenshot.png"
# Load the model
model = SentenceTransformer("clip-ViT-L-14")
# loading the image
image = Image.open(original_image)
image2 = Image.open(copied_image)
print("Original image size: ", image.size)
print("Scrrenchot image size: ", image2.size)
# computing the embeddings
emb = model.encode(image,convert_to_tensor=True)
emb2 = model.encode(image2,convert_to_tensor=True)
print(type(emb))
print(type(emb2))

# TODO Loop through image folder and find all images 
# TODO Store embeddings in a list with it's corresponding images path
# TODO store the embeddings from list to a database
# TODO use caching to store and work with temp memory
# TODO group similar images together
# TODO Notify user of similar images and propt for further action
# TODO Further action: Delete duplicates automatically or make user select original photos

# Checking for similarity in tensor 
similarities = model.similarity(emb,emb2)
print(similarities)


# using clustering algo[k means] to cluster the tensors together:
kmean = KMeans



