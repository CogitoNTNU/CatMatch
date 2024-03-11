import math
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

from catmatch.server.utils import load_image_paths

# Load the trained model (make sure to provide the path to your model's weights)
embedding_model = models.efficientnet_b1(pretrained=False)
embedding_model.eval()  # Set the model to evaluation mode

# If you used the standard ResNet50, it includes a final fully connected layer for classification.
# Assuming you want embeddings from the layer before that, you'll need to remove this layer.
# We will modify the model to remove the last fully connected layer.
embedding_model = torch.nn.Sequential(*(list(embedding_model.children())[:-1]))

# Define the image transforms
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_model(file_weights_path="./checkpoints/best.ckpt"):
    # Load the pre-trained EfficientNet B1 model
    model = models.efficientnet_b1(pretrained=False)
    model.load_state_dict(torch.load(file_weights_path))
    model.classifier = torch.nn.Sequential(torch.nn.Identity())

    # Set model to evaluation mode
    model.eval()

    return model


def get_image_embeddings_single_image(image):
    model = prepare_model()
    # Preprocess the image
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)

    # Get the embeddings from the model
    with torch.no_grad():
        embeddings = model(batch_t)

    return embeddings


def get_image_embeddings(image_folder, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = prepare_model()
    model.to(device)

    image_paths = load_image_paths()
    files = [
        Path(image_path.replace(".data/", image_folder)) for image_path in image_paths
    ]

    session = new_session("u2net")

    n_batches = math.ceil(len(files) / batch_size)
    embeddings = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_batches), desc="image batches"):
            batch_paths = files[i * batch_size : (i + 1) * batch_size]
            batch = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image = remove(image, session=session).convert("RGB")
                    batch.append(preprocess(image))
                except Exception:
                    print(f"error in reading file {path.as_posix()}")

            if len(batch) == 0:
                continue
            batch = torch.stack(batch).to(device)

            batch_embeddings = model(batch).detach().cpu().numpy()
            embeddings += [embedding for embedding in batch_embeddings]

    return np.array(embeddings)


def calculate_similarity_matrix(embeddings: np.ndarray):
    return cosine_similarity(embeddings, embeddings)


def save_similarity_matrix(
    similarity_matrix: np.ndarray, filename="similarity_matrix_effnet.hdf5"
):
    f = h5py.File(filename, "w")

    f.create_dataset("data", data=similarity_matrix)


def main():
    embeddings = get_image_embeddings("/home/ulrikro/datasets/CatBreeds/")
    similiarity_matrix = calculate_similarity_matrix(embeddings)
    similiarity_matrix = similiarity_matrix.astype(
        np.float16
    )  # Reduce precision for lower file size
    save_similarity_matrix(similiarity_matrix)


if __name__ == "__main__":
    main()
