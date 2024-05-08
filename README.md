
# Smart Math Canvas: Interactive Equation Detection and Solution Prediction

Smart Math Canvas utilizes computer vision and machine learning techniques to recognize and solve handwritten mathematical equations in real-time. This project provides an interactive and intuitive interface for users to input mathematical expressions by writing them in the air, which are then instantly recognized and solved by the system.

## Features

- **Air-Writing Recognition:** Users can generate mathematical expressions by writing them in the air using a pointer.
- **Real-time Equation Detection:** The system uses computer vision techniques to detect and extract handwritten equations in real-time.
- **CNN Model:** A Convolutional Neural Network (CNN) model is employed to recognize and classify the individual components of the equations.
- **Solution Prediction:** The system evaluates the recognized expressions to calculate their solutions.
- **Interactive Canvas:** Users can draw mathematical expressions on a canvas interface.
- **Gesture Detection:** Gesture detection is implemented to recognize when the user intends to calculate an expression.

### Architecture Details

The CNN architecture consists of the following layers:

1. **Input Layer:** 
   - The input shape is set to (40, 40, 1), corresponding to the size of the input images (40x40 grayscale).

2. **Convolutional Layers:**
   - Two convolutional layers with 32 and 64 filters, respectively, and ReLU activation functions are used to extract features from the input images.

3. **MaxPooling Layers:**
   - MaxPooling layers with a pool size of (2, 2) are added after each convolutional layer to downsample the feature maps.

4. **Flatten Layer:**
   - The Flatten layer is used to flatten the output from the convolutional layers into a one-dimensional vector.

5. **Dense Layers:**
   - Two fully connected Dense layers with 128 and 16 units, respectively, and ReLU and softmax activation functions are added to perform classification.

### Training Parameters

- **Optimizer:** Adam optimizer with a learning rate of 5e-4 is used for optimization.
- **Loss Function:** Categorical cross-entropy loss function is utilized for multi-class classification.
- **Metrics:** Accuracy is chosen as the evaluation metric to monitor model performance during training.


## How It Works

1. **Model Training:**
   - The CNN model is trained on a combined dataset of mathematical operators and MNIST digits in a Colab environment.
   - The trained model architecture and weights are saved to files (`my_model_architecture.json` and `my_model_weights.h5`).

2. **Main Script (`main.py`):**
   - The script utilizes OpenCV for computer vision tasks and implements the functionality for the Smart Math Canvas.
   - Users can write mathematical expressions in the air using a pointer, which are detected and extracted in real-time.
   - Gesture detection is implemented to trigger the calculation process when the user intends to calculate an expression.
   - Contours are created from the screenshot of the canvas, and each contour is passed through the CNN model to predict the corresponding mathematical symbol.
   - The predicted symbols are concatenated to form the complete equation, which is then passed to the `sympy` library for evaluation.
   - Exception handling is implemented to handle scenarios like division by zero or other errors during equation solving.

## Requirements

- Python 3.x
- OpenCV
- Keras 2.12
- TensorFlow 2.12
- imutils
- sympy

## Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `main.py` script to start the Smart Math Canvas application.
4. Follow the on-screen instructions to interact with the application.

## Acknowledgements

- This project was inspired by the advancements in computer vision and machine learning.
- The implementation utilizes techniques and libraries such as OpenCV, Keras, TensorFlow, and sympy.
- Special thanks to the developers of imutils for providing useful contour processing functions.
