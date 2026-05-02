import os
import pandas as pd

def create_csv_from_folders(base_path, folder_name):

    data = []
    full_path = os.path.join(base_path, folder_name)
    categories = ["NORMAL", "PNEUMONIA"]

    for category in categories:
        path = os.path.join(full_path, category)
        label = 0 if category == "NORMAL" else 1

        if os.path.exists(path):
            for img in os.listdir(path):
                if img.lower().endswith(".jpeg"):
                    img_path = os.path.join("data", folder_name, category, img)
                    data.append([img_path, label])

                else:
                    print(f"Папка {path} не найдена")
    

    df = pd.DataFrame(data, columns=["image_path", "label"])
    output_filename = f"{folder_name}_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Файл {output_filename} создан. Найдено изображений: {len(df)}")


if __name__ == "__main__":

    base_data_path = ""
    subsets = ["train", "test", "val"]

    for subset in subsets:
        create_csv_from_folders(base_data_path, subset)