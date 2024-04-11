from net import KuzushijiClassifier
from sklearn.model_selection import train_test_split
from transform import *
from engine import *
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def visualize_results(image_path, predictor, labels, drawing_font, device):
    source_image = cv2.imread(image_path)
    original_img = Image.open(image_path)
    drawing_context = ImageDraw.Draw(original_img)
    grey_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(grey_image, 130, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    identified_characters = []
    for bbox in bounding_boxes:
        dimension = int(bbox[3] * 1.6)
        y_center = int(bbox[1] + bbox[3] // 2 - dimension // 2)
        x_center = int(bbox[0] + bbox[2] // 2 - dimension // 2)
        region_of_interest = threshold_image[y_center:y_center + dimension, x_center:x_center + dimension]
        if region_of_interest.size > 7000:
            cv2.rectangle(source_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (225, 0, 0), 6)
            resized_roi = cv2.resize(region_of_interest, (100, 100))
            _, binary_image = cv2.threshold(resized_roi, 155, 255, cv2.THRESH_BINARY)
            # Preparing image for prediction
            prepared_image = np.expand_dims(binary_image, axis=0)
            prepared_image = np.expand_dims(prepared_image, axis=0)
            prepared_image_tensor = torch.tensor(prepared_image, dtype=torch.float).to(device)
            prediction = predictor(prepared_image_tensor)
            predicted_label = torch.argmax(prediction, axis=1)
            character = labels.inverse_transform(predicted_label.cpu().numpy())
            identified_characters.append(str(character[0]))
            drawing_context.text((bbox[0] + 10, bbox[1]), str(character[0]), fill=(0, 22, 225, 0), font=drawing_font)
    return source_image, original_img




train_csv_path = "./Datasets/train.csv"
unicode_csv = "./Datasets/unicode_translation.csv"
image_path_for_visualization = "./Datasets/train_images/100241706_00005_1.jpg"
# train_csv_path = "D:\\project\\kuzushiji-recognition\\train.csv"
# unicode_csv = "D:\\project\\kuzushiji-recognition\\unicode_translation.csv"

df_train = pd.read_csv(train_csv_path)

# img, annotation = df_train.values[np.random.randint(len(df_train))]
# annotate_image(image_path_for_visualization, unicode_csv, annotation)

np.random.seed(1337)

extracted_images, labels_list = extract_characters_and_labels(df_train, unicode_csv)
print(extracted_images.shape)
unique, counts = np.unique(labels_list, return_counts=True)
print(unique, counts)
NC = len(unique)

images_tensor, labels_tensor = preprocess_images_and_labels(extracted_images, labels_list, 100, 100)
images_train, images_val, labels_train, labels_val = train_test_split(images_tensor, labels_tensor, random_state=42)
images_train = images_train.permute(0, 3, 1, 2)
images_val = images_val.permute(0, 3, 1, 2)

model = KuzushijiClassifier(NC).cuda()
summary(model, input_size=(1, 100, 100))

print(images_train.shape)
print(labels_train.shape)

labels_train_indices = np.argmax(labels_train, axis=1)
labels_val_indices = np.argmax(labels_val, axis=1)

labels_train_indices = torch.tensor(labels_train_indices, dtype=torch.long)
labels_val_indices = torch.tensor(labels_val_indices, dtype=torch.long)

train_dataset = TensorDataset(images_train, labels_train_indices)
val_dataset = TensorDataset(images_val, labels_val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(model, train_loader, val_loader,
                                                                            criterion, optimizer, num_epochs=40)
model_path = 'model_Kuzushiji.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extracted_images, labels_list = extract_characters_and_labels(df_train, unicode_csv)

# visualize the data
plt.figure()
plt.imshow(extracted_images[90])
plt.axis('off')
#plt.show()
plt.savefig("visualize_one_image.png", bbox_inches='tight', dpi=300)

print(labels_list[90])

unique, counts = np.unique(labels_list, return_counts=True)
NC = len(unique)

labels = LabelEncoder()
y_integer = labels.fit_transform(labels_list)

model = KuzushijiClassifier(NC).cuda()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model.eval()

font = ImageFont.truetype('NotoSansCJKjp-Regular.otf', 50, encoding='utf-8')
image_path = "./Datasets/test_images/test_4e5458ac.jpg"
#image_path = "D:\\project\\kuzushiji-recognition\\test_images\\test_4e5458ac.jpg"

img, imsource = visualize_results(image_path, model, labels, font, device)

plt.figure(figsize=(60, 60))

plt.subplot(1, 4, 1)
plt.title("Detection of Kuzushiji", fontsize=30)
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Recognition of Kuzushiji", fontsize=40)
plt.imshow(imsource)
plt.axis('off')

plt.savefig("kuzushiji_detection_recognition.png", bbox_inches='tight', dpi=300)

#plt.show()




