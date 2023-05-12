import os  # operating systems module, used for creating directories


data_set_name = "sign_language_abc"

classes = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z"
]

classes_formatted = [
    class_name.replace(' ', '_').lower() for class_name in classes
]

classes_sorted = list(sorted(classes_formatted))

print(f"Making folder for dataset {data_set_name}.")
os.makedirs(data_set_name, exist_ok=True)
subfolders = [
    "train", "test", "val"
]
for subfolder in subfolders:
    subfolder = os.path.join(data_set_name, subfolder)
    print(f"\tCreating subfolder {subfolder}.")
    os.makedirs(subfolder, exist_ok=True)
    for class_name in classes_sorted:
        class_folder = os.path.join(subfolder, class_name)
        print(f"\t\tCreating class folder {class_folder}")
        os.makedirs(class_folder, exist_ok=True)
labels_file = os.path.join(data_set_name, 'labels.txt')
print(f"Creating labels file: {labels_file}")
with open(labels_file, 'wt') as out_file:
    for class_name in classes_sorted:
        out_file.writelines(class_name+"\n")
