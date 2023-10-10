import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fundus OCT Challenge")
    parser.add_argument("-f", "--folder", required=True, help="Path to the folder with data.", type=str)
    parser.add_argument("-r", "--ratio", default=0.2, help="Validation set ratio.", type=float)
    parser.add_argument("-s", "--seed", default=None, help="Random seed.")
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, help="Enable stratification.")
    args = parser.parse_args()


    root_dir = args.folder

    # Initialize empty lists to store image file names and their corresponding class labels
    image_names = []
    labels = []

    # Loop through each subdirectory (class) in the train folder
    for class_folder in os.listdir(os.path.join(root_dir, 'train')):
        class_folder_path = os.path.join(root_dir, 'train', class_folder)
        image_names.extend(os.listdir(class_folder_path))
        labels.extend([class_folder] * len(os.listdir(class_folder_path)))
        

    # Create a DataFrame from the image_names and labels lists
    df = pd.DataFrame({'image_name': image_names, 'label': labels})
    if args.stratify:
        train_df, val_df = train_test_split(df, test_size=args.ratio, stratify=df['label'], random_state=args.seed)
    else:
        train_df, val_df = train_test_split(df, test_size=args.ratio, random_state=args.seed)

    # Save the train and validation CSV files
    train_df.to_csv(os.path.join(root_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(root_dir, 'val.csv'), index=False)

    print("Train and validation CSV files created successfully.")
