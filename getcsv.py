import os

cifar_root = "cifar"

csv_files = {}

try:
    for root, dirs, files in os.walk(cifar_root):
        for file in files:
            if file.endswith(".jpg"):
                parts = root.split(os.sep)
                if len(parts) > 2:
                    category = parts[-1]
                    dataset_type = parts[-2]
                    filename = file
                    csv_filename = os.path.join(cifar_root, f"{dataset_type}.csv")
                    if csv_filename not in csv_files:
                        csv_files[csv_filename] = open(csv_filename, "w")
                    csv_files[csv_filename].write(f"{filename},{category}\n")
finally:
    for f in csv_files.values():
        f.close()
