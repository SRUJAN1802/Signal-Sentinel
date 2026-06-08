**Radio Link Degradation Prediction**



**About the Project**



This project is about detecting radio link degradation in cellular networks using machine learning.

The model checks network parameters and predicts whether the network condition is Normal or Degraded.



The idea is to detect problems early instead of waiting for complete network failure.



**Problem Statement**



In mobile networks, poor signal conditions can cause delays, packet loss, and bad user experience.

If network degradation is detected early, corrective actions can be taken.



This project helps to:



Analyze network signal data



Identify whether the network is normal or degraded



Provide quick predictions through a simple web app



**Dataset**



The dataset contains simulated network data such as:



Signal Strength (dBm)



Signal Quality (%)



Latency (ms)



Network Type (3G, 4G, LTE, 5G)



Since extreme failure cases are very rare, the project is treated as a binary classification problem.



**Features Used**



Signal Strength



Signal Quality



Latency



Network Type



**Model Used**



Random Forest Classifier



This model was chosen because:



It works well with tabular data



It gives good accuracy



It is simple and reliable



**Project Flow**



Load the dataset



Preprocess and prepare features



Split data into training and testing sets



Train the machine learning model



Evaluate the model



Predict network condition



Deploy using Streamlit



**Web Application**



A Streamlit web app is used to:



Enter network values



Check the network condition



See results instantly



Output:



🟢 Normal



🟡 Degraded



**Evaluation**



The model performance is checked using:



Accuracy



Precision



Recall



Confusion Matrix



The model gives stable results for new input values.



**How to Run the Project**

1️⃣ Create Virtual Environment

conda create -n rlf\_ui python=3.10

conda activate rlf\_ui



**Conclusion**



This project shows how machine learning can be used to monitor network quality and detect degradation early in a simple and effective way.





