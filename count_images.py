import os
import csv

root_folder = r"C:\fruit_dataset"



image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

with open("dataset_counts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Folder", "Image_Count"])

    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)

        if os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):

                count = sum(
                    1 for file in files
                    if file.lower().endswith(image_extensions)
                )

                if count > 0:
                    writer.writerow([
                        dataset,
                        os.path.relpath(root, dataset_path),
                        count
                    ])

print("dataset_counts.csv created successfully")