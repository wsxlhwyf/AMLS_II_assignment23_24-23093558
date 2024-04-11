# README -- AMLS_assignment_kit_23-24
The chosen challenge is "Kuzushiji Recognition" from Kaggle. <br>
This challenge is released 4 years ago and has nearly 300 teams participate in it. <br>
This challenge is solved by using Pytorch to build a Convolutional Neural Netwokr (CNN).<br>

# The role of each file
1. "NotoSansCJKjp-Regular.otf" is used to visualize the Kuzushiji and model Japanese characters.<br>
2. 'main.py' file is used to select whether to choose to run Task A or Task B (They have same algorithm with different size of dataset):<br>
3. Folder "A": Task A (smaller datasize:500. shorter running time:40mins) <br>
4. Folder "B": Task B (whole datasize:3605. longer running time:18hrs). <br>
5. Folder "Datasets": contains the datasets for this programme。<br>

# The dataset for this programme
The datasets are downloaded from the website Kaggle: https://www.kaggle.com/competitions/kuzushiji-recognition/data<br>
Since the datasize is too large, it can not upload to github.<br>
After downloading the data from the website, the data needs to store in the "Datasets" folder.<br>
The format should as follow:<br>
&emsp;>Datasets<br>
&emsp;&emsp;>test_images (file to store all the test images)<br>
&emsp;&emsp;>train_images (file to store all the train images)<br>
&emsp;&emsp;>train.csv<br>
&emsp;&emsp;>unicode_translation.csv<br>

# The environment of both tasks
1. The environment of running this programme needs to have the Pytorch and the method of downloading the Pytorch is in the website: https://pytorch.org/<br>
2. Moreover, to run this program, several libraries need to be import: numpy, matplotlib, sklearn, torch, cv2, PIL, Pandas and tqdm<br>

# Task A
1. Task A has a smaller datasize:500 and a shorter running time:40mins, it can be read by running the "main.py"<br>
2. Model will be stored in the "model_Kuzushiji.pt", and 3 graphs will be created: "visualize_onee_image.png", "kuzushiji_detection_recognition.png" and "training_history.png"<br>
3. There are 4 python file in this folder: "A.py", "engine.py", "net.py" and "transform.py"<br>
4. "A.py" is the main module (connect "engine.py", "net.py" and "transform.py")<br>
5. "engine.py" contains the training and validation process, with early stop and plotting loss&acc graphs<br>
6. "net.py" is how to create the CNN model<br>
7. "transform" is the data-processing<br>

# Task B
1. Task B has the whole datasize:3605 and the whole running time:18hrs with GPU, it can be read by running the "main.py"<br>
2. Model will be stored in the "model_Kuzushiji.pt", and 3 graphs will be created: "visualize_onee_image.png", "kuzushiji_detection_recognition.png" and "training_history.png"<br>
3. There are 4 python file in this folder: "B.py", "engine.py", "net.py" and "transform.py"<br>
4. "A.py" is the main module (connect "engine.py", "net.py" and "transform.py")<br>
5. "engine.py" contains the training and validation process, with early stop and plotting loss&acc graphs<br>
6. "net.py" is how to create the CNN model<br>
7. "transform" is the data-processing <br> 
