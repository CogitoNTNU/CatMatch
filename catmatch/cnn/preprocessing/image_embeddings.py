import torch
from PIL import Image
from torchvision import models, transforms
import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import h5py

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


def get_image_embeddings(image_folder, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = prepare_model()
    model.to(device)

    image_folder = Path(image_folder)
    files = list(image_folder.rglob("**/*.*"))  # Get all files recursively, no folders

    n_batches = int(torch.ceil(len(files) / batch_size))
    embeddings = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_batches), prefix="batch"):
            batch_paths = files[i * batch_size, (i + 1) * batch_size]
            batch = []

            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                batch.append(image)

            batch_preprocessed = preprocess(batch).to(device)

            batch_embeddings = model(batch_preprocessed).detach().cpu().numpy()
            embeddings += [embedding for embedding in batch_embeddings]

    return np.array(embeddings)


def calculate_similarity_matrix(embeddings: np.array):
    return cosine_similarity(embeddings, embeddings)


def save_similarity_matrix(
    similiarity_matrix: np.array, filename="similiary_matrix.hdf5"
):
    f = h5py.File(filename, "w")

    f.create_dataset("data", data=similiarity_matrix)


def main():
    embeddings = get_image_embeddings("datasets/images")
    similiarity_matrix = calculate_similarity_matrix(embeddings)
    save_similarity_matrix(similiarity_matrix)


if __name__ == "__main__":
    main()
