import torch
from PIL import Image
from torchvision import models, transforms

# Load the trained model (make sure to provide the path to your model's weights)
embedding_model = models.resnet50(pretrained=False)
embedding_model.load_state_dict(torch.load("./checkpoints/best.ckpt"))
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


def get_image_embeddings(image: Image.Image):
    preprocessed = preprocess(image)
    preprocessed = preprocessed.unsqueeze(0)
    with torch.no_grad():
        embeddings = embedding_model(preprocessed)
        embeddings = torch.flatten(embeddings, 1)
        return embeddings


def main():
    image = Image.open("./.data/Abyssinian/Abyssinian-35021741_239.jpg").convert("RGB")
    # show image
    image.show()
    embeddings = get_image_embeddings(image)
    print("Shape: ", embeddings.shape)
    print("Embeddings: ", embeddings)


if __name__ == "__main__":
    main()
