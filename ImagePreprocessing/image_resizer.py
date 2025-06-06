import os
from PIL import Image

def resize_dataset(dataset_root, target_size=(400, 400)):
    subsets = ["train", "val"]
    classes = ["Normal", "DDoS"]

    for subset in subsets:
        for cls in classes:
            class_path = os.path.join(dataset_root, subset, cls)
            if not os.path.exists(class_path):
                print(f"Skipping missing folder: {class_path}")
                continue

            seq_folders = [d for d in os.listdir(class_path)
                           if d.startswith("seq_") and
                           os.path.isdir(os.path.join(class_path, d))]
            for seq_folder in seq_folders:
                seq_path = os.path.join(class_path, seq_folder)
                png_files = [f for f in os.listdir(seq_path)
                             if f.endswith(".png")]
                for png_file in png_files:
                    png_path = os.path.join(seq_path, png_file)
                    with Image.open(png_path) as img:
                        resized_img = img.resize(target_size, Image.LANCZOS)
                        resized_img.save(png_path)

if __name__ == "__main__":
    dataset_root = "C:\\Users\\cybears\\Downloads\\DDoS\dataset_root"
    resize_dataset(dataset_root, (400, 400))
    print("All images have been resized to 400x400.")
