NeuroMove HAR: Human Activity Recognition using LSTM and Streamlit

NeuroMove HAR is a human activity recognition system developed using deep learning and smartphone sensor data. The system classifies six daily physical activities including walking, walking upstairs, walking downstairs, sitting, standing and laying. The project combines an LSTM based machine learning model for activity prediction with an interactive Streamlit dashboard for visualization and performance evaluation.

Overview

The objective of this project is to automatically classify physical activities from sensor readings collected by a smartphone placed at the waist. The dashboard supports single sample classification, batch evaluation of datasets and simulated live streaming of sensor inputs. The interface also provides biomechanical visualization using a 3D skeleton model and waveform plots that represent accelerometer and gyroscope patterns for each prediction.

Dataset

The project uses the UCI Human Activity Recognition Using Smartphones dataset. The dataset contains 561 preprocessed statistical features extracted from inertial sensor data captured from 30 volunteers performing daily activities. The dataset includes training and testing splits and does not require relabeling.

Dataset source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Model

The classification model is built using an LSTM neural network. The final architecture contains an LSTM layer with 128 units, followed by a dense hidden layer with 64 neurons and ReLU activation, and a softmax output layer with six activity categories. Dropout regularization is applied to reduce overfitting, and the model is trained using the Adam optimizer and sparse categorical crossentropy loss.

Final performance on the test dataset:
Test Accuracy: 94.03%
Test Loss: 0.1848

The model outputs the predicted activity and a confidence score for each sample.

Features of the Dashboard

• Single sample prediction with biomechanics visualization

• Batch analysis mode with running accuracy and confusion matrix

• Simulated live stream of sensor data for demonstration

• Confidence gauge and waveform visualization of accelerometer and gyroscope signals

• Sensor contribution analysis for interpretability

• Automatic text report generation for predictions

Installation and Execution

Clone the repository

git clone https://github.com/yourusername/NeuroMove-HAR.git

cd NeuroMove-HAR

Install dependencies

pip install -r requirements.txt

Add the dataset

Download and extract the UCI HAR Dataset folder into the project directory so the structure appears as:

NeuroMove-HAR / UCI HAR Dataset

(Optional) Retrain the model

python train_model.py

Run the application

streamlit run app.py

Future Scope

• Integration with ESP32 or Arduino IMU sensors for real time wearable sensing

• Bluetooth support for streaming smartphone and smartwatch sensor data

• Mobile application for physiotherapy monitoring and elderly movement tracking

• Cloud connectivity for long term motion pattern analysis and alerting systems

• Edge deployment for wearable devices with low power inference
