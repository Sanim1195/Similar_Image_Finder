from PIL import Image
import torch
import numpy as np
import cv2 as cv
import clip

# read image
# resize image maintaining the aspect ratio
# normalise the resized image
# get the embeddings
# check for similarity with cosine similarity and euclidean distance


device = "cpu"
image_path = "assets/photos/original.jpg"
image_path2 = "assets/photos/original.jpg"

# Load CLIP model and preprocess function
model, preprocess = clip.load("ViT-L/14", device, jit=False)

# Preprocess the images and add batch dimension
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

print("The data type of images after being processed is: ", type(image))

def calculate_distance(embedding1,embedding2):
    """ Calculates Euclidean Distance between 2 images """
    dist = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    return(dist)

def calculate_cosine_similarity(embedding1, embedding2):
    """ Calculates Cosine Similarity between 2 image embeddings """
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)


def resize_with_aspect_ratio(raw_image_path, new_width):
    # Read the original image
    original_image = cv.imread(raw_image_path)

    # Calculate the aspect ratio
    aspect_ratio = original_image.shape[1] / original_image.shape[0]

    # Determine the new height based on the desired width
    calculated_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = cv.resize(original_image, (new_width, calculated_height))
    print("Newly Resized images shape: " , resized_image.shape)
    cv.imshow("Screen", resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return resized_image


def normalize(image):
    # Compute the standard deviation for each channel
    channel_stds = np.std(image, axis=(0, 1))
    print(channel_stds)
    # Normalize by dividing image by the channel standard deviations
    normalized_image = image  / channel_stds
    return(normalized_image)


# Get image embeddings from the model
with torch.no_grad():
    emb1 = model.encode_image(image)
    emb2 = model.encode_image(image2)

print("The data type of embeddings are: ", type(emb1) , "And ", type(emb2))

# Calculate cosine similarity between embeddings
similarity = calculate_cosine_similarity(emb1, emb2)
distance = calculate_distance(emb1,emb2)

print(f"Cosine Similarity: {similarity.item()}  {type(similarity)}")
print(f"Euclidean Distance: {distance.item()} {type(similarity)}")






# using clustering algo[k means] to cluster the tensors together:
# kmean = KMeans



# TODO Loop through image folder and find all images 
# TODO Store embeddings in a list with it's corresponding images path
# TODO store the embeddings from list to a database
# TODO use caching to store and work with temp memory
# TODO group similar images together
# TODO Notify user of similar images and propt for further action
# TODO Further action: Delete duplicates automatically or make user select original photos
