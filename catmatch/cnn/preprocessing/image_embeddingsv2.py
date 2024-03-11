import math
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel

from catmatch.server.utils import load_image_paths

# embedding_model = models.efficientnet_b1(pretrained=False)
# embedding_model.eval()  # Set the model to evaluation mode


# Load the trained model (make sure to provide the path to your model's weights)
# If you used the standard ResNet50, it includes a final fully connected layer for classification.
# Assuming you want embeddings from the layer before that, you'll need to remove this layer.
# We will modify the model to remove the last fully connected layer.


preprocess = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
    ]
)
# Load the feature extractor and model from Hugging Face


def get_image_embeddings(image_folder, batch_size=128) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    session = new_session("u2net")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.to(device)

    image_paths = load_image_paths()
    files = [
        Path(image_path.replace(".data/", image_folder)) for image_path in image_paths
    ]

    n_batches = math.ceil(len(files) / batch_size)
    embeddings = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_batches), desc="image batches"):
            batch_paths = files[i * batch_size : (i + 1) * batch_size]
            batch = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    no_background = remove(image, session=session).convert("RGB")
                    batch.append(preprocess(no_background))
                except Exception:
                    print(f"error in reading file {path.as_posix()}")

            if len(batch) == 0:
                continue
            batch = torch.stack(batch).to(device)
            inputs = processor(images=batch, return_tensors="pt")
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)

            output = model(**inputs)
            batch_embeddings = output.last_hidden_state.detach().cpu().numpy()
            embeddings += [embedding for embedding in batch_embeddings]

    return np.array(embeddings)


def calculate_similarity_matrix(embeddings: np.ndarray):
    return cosine_similarity(embeddings, embeddings)


def save_similarity_matrix(
    similarity_matrix: np.ndarray, filename="similarity_matrix.hdf5"
):
    f = h5py.File(filename, "w")

    f.create_dataset("data", data=similarity_matrix)


def main():
    # embeddings = get_image_embeddings("/home/ulrikro/datasets/CatBreeds/")
    embeddings: np.ndarray = np.load("embeddings.npy")
    print("embeddings.shape", embeddings.shape)
    # np.save("embeddings.npy", embeddings)
    # Flatten
    embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)

    print("embeddings_2d.shape", embeddings_2d.shape)
    similiarity_matrix = calculate_similarity_matrix(embeddings_2d)
    similiarity_matrix = similiarity_matrix.astype(
        np.float16
    )  # Reduce precision for lower file size
    save_similarity_matrix(similiarity_matrix, "similarity_matrix_dinov2_flatten.hdf5")


if __name__ == "__main__":
    main()
