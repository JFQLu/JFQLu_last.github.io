##### Project Collaborators: *Andrew William, Dan Visitchaichan, Daniel Dinh, Michael Li*

### Project Details
This project involved collaborative work with 4 other data scientists over 10 weeks where we defined the problem, explored and transformed our data, researched solutions, implemented machine learning models to solve the probelm outlined and presented our deliverables coordinators and other data science peers. The raw data was provided by CyAmast.

### Background
The Internet of Things (IoT) is a technology that connects various devices, machines, and systems to a single network. It is widely used in many industries because it allows for efficient data gathering. However, these devices often have limited functions and are vulnerable to cyber attacks such as DDoS and MITM, making it important to have protection measures in place, such as network anomaly detection.

### Data
The data provided contained time series network data including packet/byte counts in/out of a number of ports of a number of devices. Below is a snapshot:
```python
data.info()
```
![image](https://user-images.githubusercontent.com/98208084/209839388-429df3b8-320f-4a0d-8de2-08be9d56f2d2.png)

**Here we hypothesise if we are able to classify a new datapoint into a particular flow we are able to observe when an anomaly occurs, i.e. a sudden change in classification may indicate an anomaly occuring.**

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
|GMM, K-Means, Fuzzy C-Means|Random Forest, XGBoost, AdaBoost, SVM, MLP, Logistic Regression| 

This blog will deep dive into the one-class Gaussian Mxture Model and the multi-class Random Forest models.

### Guassian Mixture Model - Deep Dive
A Gaussian mixture model (GMM) is a probabilistic model that assumes that the data is generated from a mixture of several different Gaussian distributions. It is often used for classification tasks because it allows for the modeling of complex, multimodal distributions and can handle data with uncertainty or incomplete information.

In a GMM, each data point is assumed to belong to one of the Gaussian distributions in the mixture, and the model estimates the probability that a given data point belongs to each of the different distributions. The model can then classify a new data point based on which distribution it is most likely to belong to.

Once the GMM has been trained, it can be used to classify new data points by determining the distribution that they are most likely to belong to.

#### Preprocessing 
First any preprocessing is done including Z-score scaling and/or PCA; note that we perform experiments to determine if scaling or PCA improves model perforance.

#### Compute Optimal Cluster Number
The elbow method is a technique that is often used to determine the optimal number of clusters to use in a clustering algorithm such as a Gaussian mixture model (GMM). It is based on the idea that the optimal number of clusters is the point at which the decrease in the sum of squared distances between the points and their closest cluster centers starts to level off. 
```python
def compute_optimal_clusters(train,n_clusters,random_state):
    print("Calculating optimal number of clusters from elbow method")
    km = KMeans(random_state=random_state)
    visualizer = KElbowVisualizer(km, k=n_clusters,show=False)
    t1 = time.perf_counter()
    visualizer.fit(train)
    t2 = time.perf_counter()       
    optimal_k = visualizer.elbow_value_


    if optimal_k is None:
    optimal_k = 1

    print(f"Optimal number of clusters:{optimal_k}")
    optimal_k_time = t2-t1
    print('Time taken to find optimal clusters:',optimal_k_time,'s')

    return optimal_k,optimal_k_time
```
The above function returns us the optimal number of clusters *optimal_k*.

#### Training 
```python 
# Training a GMM using optimal number of clusters
gmm, training_time= fit_gmm(train,optimal_k,random_state = random_state,flow_type = flow_type)
```
The model is trained using sklearn's implementation of GMM. This model is then stored for testing. 

#### Testing 
Testing the GMM follows the following algorithm: 

**Input:**
Test feature data of same subsequence size as training feature data

Set of trained GMM classification models

Z-score scaler if *scale = True*

PCA scaler if *PCA = True*

**Output:**

Network flow predictions on test feature data

**for** each network flow **do**

    1. **if** *scale = True* **then**\
    
            Scale each feature from testing data using z-score scaler from training data;






### Random Forest - Deep Dive

### Experiment Layer 1 Results
Best experiment results for each one-class model for layer 1
![image](https://user-images.githubusercontent.com/98208084/209847168-c7621fd3-1a31-4775-931c-2494d184905d.png)

Best experiment results for each multi-class model for layer 1
![image](https://user-images.githubusercontent.com/98208084/209847202-760f6db3-8842-40ef-af26-1c087d29bf66.png)













