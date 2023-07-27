# Celebrity-Face-Recognition

This project aims to build a machine learning model that can recognize celebrities' faces from images. The code uses the Python programming language and various libraries, including OpenCV, NumPy, scikit-learn, and matplotlib.

<h3> Preprocessing:</h3>
1) Detect Face and Eyes: The code uses the Haar Cascade classifier from OpenCV to detect faces and eyes in an image. It processes each image to find faces, and then for each detected face, it checks if there are at least two eyes visible. If two eyes are detected, it considers the face suitable for further processing.<br>
2) Crop the Facial Region: Once a face with two eyes is detected, the code crops the facial region of the image and saves it.<br>
3) Wavelet Transform: The code applies the wavelet transform to the cropped face region, resulting in a transformed image that enhances the edges, such as the eyes, nose, and lips.
<br>
<h3>Dataset Creation:</h3>
The code processes a dataset containing images of different celebrities. It goes through each image and performs the preprocessing steps, saving the cropped facial region images in a separate folder for each celebrity.
<br>
<h3>Model Training:</h3>
The model used for recognition is a Support Vector Machine (SVM) classifier with an RBF kernel. The training data consists of the raw images and their corresponding wavelet-transformed images. The model is trained using this data, and GridSearchCV is employed to find the best hyperparameters for the SVM model.
<br>
<h3>Evaluation:</h3>
The trained SVM model is evaluated using a test dataset. The accuracy and classification report are provided to assess the model's performance.
The final model achieves good accuracy in recognizing celebrity faces. The code also contains options to try different models like Random Forest and Logistic Regression, but the SVM model with an RBF kernel performs best.
<br>
<h3>Usage:</h3>
To use this model, provide a test image with a celebrity's face, and the code will predict the celebrity's name based on the trained model.
This project can be further extended by adding more celebrities to the dataset, fine-tuning hyperparameters, and experimenting with other deep learning models to improve recognition accuracy.
