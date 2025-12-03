NeuroMove HAR is a Human Activity Recognition project that uses smartphone sensor data to identify physical movements such as walking, walking upstairs, walking downstairs, sitting, standing, and laying. The system uses a Long Short Term Memory (LSTM) deep learning model for classification and a Streamlit dashboard for viewing predictions and biomechanical visualizations.

This project is based on the UCI Human Activity Recognition Using Smartphones dataset. The dataset contains 561 numerical features generated from accelerometer and gyroscope readings. These features are processed and used as input to the LSTM model.

The model achieved the following performance after training:
Test Accuracy: 94.03 percent
Test Loss: 0.1848

Features included in this project:
• LSTM based activity classification
• Real time prediction for a single input sample
• Batch CSV prediction and accuracy calculation
• 3D skeleton animation of predicted movement
• Raw sensor waveform visualization
• Confidence score display
• Sensor contribution interpretation
• Automatic report generation for each prediction

Technologies used:
Python, TensorFlow Keras, NumPy, Pandas, Scikit Learn, Streamlit, Plotly

How to run:

Clone the project repository

Install the required Python libraries using pip

Download the UCI HAR Dataset and place it inside the project folder

Run the Streamlit app using the command: streamlit run app.py

Upload a CSV sample or batch file to view prediction results

Dashboard modes:
Single Sample mode allows prediction for one input at a time
Batch Analysis mode reads an entire CSV and displays accuracy, confusion matrix, precision, recall, and F1 score
Live Stream mode simulates continuous sensor input for demonstration

Folder structure recommended:
NeuroMove HAR
UCI HAR Dataset
models
screenshots
app.py
train_model.py
