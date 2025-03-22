# GuideWire
We've created a neural network that acts as an early warning system for your Kubernetes clusters, predicting failures before they cause major disruptions.

##The Dataset  

### How We Built It  
To train our model, we generated a synthetic dataset that mimics real-world Kubernetes failures. This dataset captures critical cluster metrics such as:  

-CPU usage
-Memory usage  
-Disk & network I/O  
-Pod restart count
-Node & pod status  

We simulated five failure scenarios, ensuring that each type has an equal number of records (5,000 per type).  

### Types of Failures We Detect  
1️ Healthy (Type 0): Normal system behavior—nothing to worry about!  
2️ Node Failure (Type 1):A node goes down, affecting workloads.  
3️ Frequent Pod Restarts (Type 2):** Misconfigurations or crashes cause pods to restart frequently.  
4️ High Resource Utilization (Type 3):** CPU and memory are maxed out, signaling potential slowdowns.  
5️ Network Failure (Type 4):** A sudden drop in network traffic, possibly due to connectivity issues.  

During training, we skipped `pod_status` and `node_status` to focus on more dynamic indicators of failure.  

##Tech Stack  
We used:  
- PyTorch– to power the neural network  
- Scikit-learn– for data processing & model evaluation  
- Pandas & NumPy– to handle and manipulate data  


##How Our Model Works  

Neural Network architecture:  

Input Layer: Feeds in all cluster metrics at once—no assumptions, just raw data.  
Hidden Layer 1 (128 neurons): Detects early signs of trouble.  
Batch Normalization: Helps stabilize fluctuating Kubernetes metrics.  
LeakyReLU Activation: Prevents learning from stopping when metrics flatline.  
Dropout (30%): Makes sure the model doesn’t fixate on a single failure pattern.  
Hidden Layer 2 (64 neurons): Starts grouping warning signals into meaningful failure categories.  
Hidden Layer 3 (32 neurons): Refines these patterns into clear failure signatures.  
Output Layer: Gives a probability score for each failure type.  



#Step 1: Install Dependencies  
First, install the required libraries:  

pip install torch torchvision scikit-learn pandas numpy

#Step 2: Clone the Repository 

git clone <repo_url>
cd <repo_directory>

#Step 3: Load the Dataset  
- The dataset is inside the dataset folder.  
- Open the Jupyter Notebook (`.ipynb`) file.  
- Update the dataset path in the code to match your system.  

#Step 4: Train & Test the Model  
- Run the notebook to train the model.  
- Test it using unseen Kubernetes cluster data to see how well it predicts failures.  


## Model Performance  

Our neural network achieves:  
- Accuracy:** 77.76%  
- Precision:** 78.31%  
- Recall:** 77.76%  
- F1 Score:** 74.56%  


---
