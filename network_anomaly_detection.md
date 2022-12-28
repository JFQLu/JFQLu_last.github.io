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

### Exploratory Data Analysis
Our team began by analysing and understanding the provided data. Python was used to calculate statistics and Matplotlib and seaborn packages were utilised to visualise the shape and trends of the time series data. Key observations include: 
- There was high correlation between corresponding in and out flows of each port
- There were distinct characteristics to each flow type (this is what we want for accurate classification)



