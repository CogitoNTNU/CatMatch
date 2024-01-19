import json
from pathlib import Path


def main():
    image_paths = []
    image_path = Path("./.data/")
    for folder in image_path.iterdir():
        for image in folder.iterdir():
            image_paths.append(image.as_posix())
    print(image_paths)
    with open("./image_paths.json", "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    main()
