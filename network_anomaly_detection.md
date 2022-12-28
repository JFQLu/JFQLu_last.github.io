##### Project Collaborators: *Andrew William, Dan Visitchaichan, Daniel Dinh, Michael Li*

### Project Details
This project required collaborative work with 4 other data scientists over 10 weeks where we defined the problem, explored and transformed our data, researched solutions, implemented machine learning models to solve the probelm outlined and presented our deliverables coordinators and other data science peers. The raw data was provided by CyAmast.

### Background
The Internet of Things (IoT) is a technology that connects various devices, machines, and systems to a single network. It is widely used in many industries because it allows for efficient data gathering. However, these devices often have limited functions and are vulnerable to cyber attacks such as DDoS and MITM, making it important to have protection measures in place, such as network anomaly detection.

### Data
The data provided contained time series network data including packet/byte counts in/out of a number of ports of a number of devices. Below is a snapshot:
```python
data.info()
```
![image](https://user-images.githubusercontent.com/98208084/209839388-429df3b8-320f-4a0d-8de2-08be9d56f2d2.png)

**Here we hypothesise if we are able to classify a new datapoint into a particular flow we are able to observe when an anomaly occurs, i.e. a sudden change in classification may indicate an anomaly occuring**

### Exploratory Data Analysis
Our team began by analysing and understanding the provided data. Python was used to calculate statistics and Matplotlib and seaborn packages were utilised to visualise the shape and trends of the time series data. Key observations include: 
- There was high correlation between corresponding in and out flows of each port
- There were distinct characteristics to each flow type (this is what we want for accurate classification)

### Feature Engineering
Our solution begins with the feature extraction step. This involved transforming raw training and testing data (provided by CyAmast) into training and testing feature datasets for subsequence sizes of 16, 32, 64, and 128. Note, we define a subsequence as a sequence of contiguous data from the original dataset which is transformed into a single instance in the feature datasets. The feature dataset contains 8 features for each labelled flow subsequence: average packet and byte counts for 1-, 2-, 4-, and 8-minute time frames. Furthermore, all observations which have only 0s in their 8 features were dropped from the dataset. 

A key function used in this process was the smoothen() function which allowed us to create higher time-frame data for extracting features:
```python
def smoothen(data,subsequence_length = 16,n=2):
    if len(data.shape) != 3:
      print("Subsequence is of wrong shape for our purposes, should be 3 dimensional --> (n_observations, subsequence size, n_original_features).") 

    # Makes an instance of a flow more coarse by n times.
    #print(f"length of flow instance:{len(subsequence)}")
    temp = np.split(data, subsequence_length/n,axis=1)
    temp = np.array(temp).sum(axis=-2)
    return temp
```

### Model Training
For this project we wanted to compare the effectiveness and logistics of both single-class and multi-class machine learning models in classifying network flows. For this reason we tackled this probelm with both single- and multi-class models. 
|Single-class| multi-class|
| ---------- | ---------- |
|GMM|Random Forrest| 












