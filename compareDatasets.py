import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to process class names
def process_class_name(class_name):
    # Lowercase and convert to regex-friendly format
    return re.sub(r'\s+', '_', class_name.lower())

# Function to compute similarity matrix
def compute_similarity_matrix(classes_a, classes_b):
    # Encode the class names
    embeddings_a = model.encode([process_class_name(ca) for ca in classes_a])
    embeddings_b = model.encode([process_class_name(cb) for cb in classes_b])
    
    # Compute the similarity matrix
    similarity_matrix = np.zeros((len(classes_a), len(classes_b)))
    for i, emb_a in enumerate(embeddings_a):
        for j, emb_b in enumerate(embeddings_b):
            similarity_matrix[i, j] = 1 - cosine(emb_a, emb_b)
    
    return similarity_matrix

# Function to plot similarity matrix
def plot_similarity_matrix(similarity_matrix, classes_a, classes_b, dataset_pair_name):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(classes_a)), labels=classes_a)
    plt.xticks(ticks=np.arange(len(classes_b)), labels=classes_b, rotation=90)
    plt.title(f'Similarity Matrix: {dataset_pair_name}')
    plt.xlabel('Classes from Evaluation Datasets (Folder B)')
    plt.ylabel('Classes from Pretraining&TrainingDatasets (Folder A)')
    plt.show()

# Directories
folder_a = 'Pretraining&TrainingDatasets'  # Update with the correct path
folder_b = 'EvaluationDatasets'            # Update with the correct path

# Iterate through datasets in both folders
for dataset_a in os.listdir(folder_a):
    dataset_a_path = os.path.join(folder_a, dataset_a)
    if os.path.isdir(dataset_a_path):
        classes_a = [f for f in os.listdir(dataset_a_path) if os.path.isfile(os.path.join(dataset_a_path, f))]
        
        for dataset_b in os.listdir(folder_b):
            dataset_b_path = os.path.join(folder_b, dataset_b)
            if os.path.isdir(dataset_b_path):
                classes_b = [f for f in os.listdir(dataset_b_path) if os.path.isfile(os.path.join(dataset_b_path, f))]
                
                # Compute the similarity matrix
                similarity_matrix = compute_similarity_matrix(classes_a, classes_b)
                
                # Plot and inspect the similarity matrix
                plot_similarity_matrix(similarity_matrix, classes_a, classes_b, f'{dataset_a} vs {dataset_b}')
