import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

"""
Annotates an image with given character annotations, including bounding boxes and text labels. 
It loads character mappings from a CSV file, draws bounding boxes around characters specified by annotations, 
and adds text labels beside these boxes. The function returns the annotated image as a NumPy array.
"""


def annotate_image(image_path, unicode_path, annotations):
    # Parsing the annotation string into a structured array
    FS = 50
    font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', FS, encoding='utf-8')
    parsed_annotations = np.array(annotations.split(' ')).reshape(-1, 5)
    unicode_map = {codepoint: char for codepoint, char in
                   pd.read_csv(unicode_path).values}
    # Loading the image and creating canvases for bounding boxes and characters
    source_image = Image.open(image_path).convert('RGBA')
    box_overlay = Image.new('RGBA', source_image.size)
    text_overlay = Image.new('RGBA', source_image.size)
    box_drawer = ImageDraw.Draw(box_overlay)  # For drawing bounding boxes
    text_drawer = ImageDraw.Draw(text_overlay)  # For drawing text annotations

    for codepoint, x, y, width, height in parsed_annotations:
        x, y, width, height = int(x), int(y), int(width), int(height)
        character = unicode_map[codepoint]  # Convert codepoint to the actual character

        # Drawing the bounding box and the character annotation
        box_drawer.rectangle((x, y, x + width, y + height), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        text_drawer.text((x + width + FS / 4, y + height / 2 - FS), character, fill=(0, 0, 255, 255), font=font)

        # Optionally, display the cropped character image
        cropped_image = source_image.crop((x, y, x + width, y + height))
        plt.figure()
        print(character)
        plt.imshow(cropped_image)
        plt.show()

    # Composing the image layers
    final_image = Image.alpha_composite(Image.alpha_composite(source_image, box_overlay), text_overlay)
    final_image = final_image.convert("RGB")  # Convert to RGB to remove alpha before saving in jpg format.
    return np.asarray(final_image)


"""
Extracts and preprocesses character images and their corresponding labels from a given dataframe and unicode mapping file. 
It crops character images from larger images based on annotations, resizes them, converts to grayscale, and applies binary thresholding. 
Returns arrays of processed images and labels.
"""


def extract_characters_and_labels(df_train, unicode_path):
    extracted_images = []
    labels_list = []
    unicode_map = {codepoint: char for codepoint, char in
                   pd.read_csv(unicode_path).values}
    # Iterating through a subset of the dataframe\
    for image_id, annotation_str in tqdm(df_train[:500].values):
        try:
            image_file_path = './Datasets/train_images/{}.jpg'.format(image_id)
            #image_file_path = 'D:\\project\\kuzushiji-recognition\\train_images/{}.jpg'.format(image_id)
            annotations = np.array(annotation_str.split(' ')).reshape(-1, 5)
            source_image = Image.open(image_file_path).convert('RGBA')

            for codepoint, x, y, width, height in annotations:
                x, y, width, height = int(x), int(y), int(width), int(height)
                character = unicode_map[codepoint]  # Convert codepoint to actual character

                # Cropping and preprocessing character image
                cropped_img = source_image.crop((x, y, x + width, y + height))
                resized_img = cropped_img.resize((100, 100))
                processed_img = np.asarray(resized_img)
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                _, binary_img = cv2.threshold(processed_img, 155, 255, cv2.THRESH_BINARY_INV)

                extracted_images.append(binary_img)
                labels_list.append(str(character))
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

    extracted_images = np.array(extracted_images)
    labels_list = np.array(labels_list)

    return extracted_images, labels_list


def preprocess_images_and_labels(images, labels, IMG_ROWS, IMG_COLS):
    # Initializing the label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Converting labels and images into PyTorch tensors
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    images_tensor = torch.tensor(images, dtype=torch.float32)

    # One-hot encoding the labels
    one_hot_labels = F.one_hot(labels_tensor).float()

    # Reshaping and normalizing the images tensor
    total_images = images_tensor.shape[0]
    reshaped_images = images_tensor.view(total_images, IMG_ROWS, IMG_COLS, 1)
    normalized_images = reshaped_images / 255.0  # Normalizing to the range [0, 1]

    return normalized_images, one_hot_labels
