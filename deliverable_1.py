import torchvision.transforms as T
import torch
import transformers, accelerate
import PIL
import requests
import os
import time
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import login
from IPython.display import display
from datasets import Dataset, Image

## Generates Token and log in to Hugging Face
def hugging_face_login():
    # hugging_face_token_file = "hugging_face_token.txt"
    # with open(hugging_face_token_file, "r") as f:
    #     for line in f:
    #         HF_TOKEN = line
    #         break

    # access_token_read = HF_TOKEN
    # login(token=access_token_read)
    from huggingface_hub import notebook_login
    notebook_login()

# Pick an image data set
def get_paths_to_images(images_directory):

  paths = []
  for file in os.listdir(images_directory):
    paths.append(file)

  return paths

# Load the dataset from the directory
def load_dataset(images_directory):
  paths_images = get_paths_to_images(images_directory)
  dataset = Dataset.from_dict({"image": paths_images})
  return dataset

# Split data into train and test
def split_dataset(dataset, test_size=0.2):
  dataset_training = dataset.train_test_split(test_size=test_size)
  return dataset_training

def generate_candidate_subset(dataset_training):
    dataset_training = split_dataset(dataset)
    candidate_subset = dataset_training["train"]
    return candidate_subset

# Load base model for image embeddings
def load_model():
    model_ckpt = "jafdxc/vit-base-patch16-224-finetuned-flower"
    extractor = transformers.AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = transformers.AutoModel.from_pretrained(model_ckpt)
    return model, extractor

# Data transformation chain.
def generate_transformation_chain(extractor):
    transformation_chain = T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize(int((256 / 224) * extractor.size["height"])),
            T.CenterCrop(extractor.size["height"]),
            T.ToTensor(),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ]
    )
    return transformation_chain

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(PIL.Image.open("img/" + image)) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp

# Compute the similarity scores
def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def fetch_similar(image, top_k=3):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    print("inside fetch_similar")
    image_transformed = transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    # Comute the embedding.
    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    sim_scores = compute_scores(all_candidate_embeddings, query_embeddings)
    similarity_mapping = dict(zip([str(index) for index in range(all_candidate_embeddings.shape[0])], sim_scores))

    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id_entries = list(similarity_mapping_sorted.keys())[:top_k]

    ids = list(map(lambda x: int(x.split("_")[0]), id_entries))
    #labels = list(map(lambda x: int(x.split("_")[-1]), id_entries))
    return ids

# Find closest flower
def find_closest_flower():
    images = []
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        uploaded_image = PIL.Image.open(uploaded_file).convert('RGB')

        sim_ids = fetch_similar(uploaded_image, 3)
        names = []
        names.append(os.path.splitext(uploaded_file.name)[0])

        for id in sim_ids:
            images.append(PIL.Image.open("img/" + candidate_subset_emb[id]["image"]))
            names.append(os.path.splitext(candidate_subset_emb[id]["image"])[0])
        
        images.insert(0, uploaded_image)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
          st.image(images[0], caption="Query Image - " + (names[0]))
        with col2:
          st.image(images[1], caption="Similar Image 1 - " + (names[1]))
        with col3:
          st.image(images[2], caption="Similar Image 2 - " + (names[2]))
        with col4:
          st.image(images[3], caption="Simlar Image 3 - " + (names[3]))
        st.write("")


if __name__ == "__main__":
    hugging_face_login()
    path_images = "img/"
    dataset = load_dataset(path_images)
    dataset_training = split_dataset(dataset)
    candidate_subset = generate_candidate_subset(dataset_training)
    model, extractor = load_model()
    transformation_chain = generate_transformation_chain(extractor)
    
    # Here, we map embedding extraction utility on our subset of candidate images.
    batch_size = 24
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device))
    candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=24)

    all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
    all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

    # Generate Streamlit
    st.title("Flower Image Search Similarity Demo")
    st.subheader("Pass in an input image file")
    find_closest_flower()


