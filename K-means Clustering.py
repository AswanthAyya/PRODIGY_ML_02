#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1.import necessary libraries
2.import dataset
data preprocessing
3.select the features to cluster
4.find optimal number of clusters---elbow method
5.train the model on the dataset using the optimal cluster k value
6.Visulalize the clusters


# ## 1.import necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 2.import dataset

# In[2]:


df=pd.read_csv("Mall_Customers.csv")
df.head()


# In[3]:


df.info()


# ## 3.select the features to cluster
# Annual Income (k$)	Spending Score (1-100)

# In[7]:


X=df.iloc[:,3:].values


# In[ ]:





# In[9]:


plt.scatter(X[:,0],X[:,1])


# ## 4.find optimal number of clusters---elbow method

# In[ ]:


WCSS=sum of the distances of observations from their cluster centroids


 k   wcss


# In[11]:


from sklearn.cluster import KMeans
#intialise the list to store wcss values
wcss=[]
# Try different values of K (from 1 to 10) and calculate WCSS for each K
#K-Means++ to Choose Initial Cluster Centroids for K-Means Clustering
for k in range(1,11): #k= 1 to 10
    kmeans=KMeans(n_clusters=k,init="k-means++",random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    


# In[12]:


wcss


# In[16]:


#plot the elbow curve
plt.figure(figsize=(8,6))
plt.plot(range(1,11),wcss,marker="o",color="red")
plt.title("Elbow method for finding optimal k")
plt.xlabel("number of clusters k")
plt.ylabel("wcss")
plt.grid()
plt.show()


# ## 5.train the model on the dataset using the optimal cluster k value

# In[17]:


kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)
#return a label for data based on their cluster
y_kmeans=kmeans.fit_predict(X)


# In[18]:


y_kmeans


# In[19]:


X


# ## 6.Visulalize the clusters

# In[22]:


#how many no.of datapoints belonging to 0 th cluster ffrom 0 th column anuual income
X[y_kmeans==0,0]


# In[30]:


#how many no.of datapoints blonging to 2 nd cluster from 1 st column spending score
X[y_kmeans==2,1]


# In[29]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],color="red",s=100,label="cluster1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],color="cyan",s=100,label="cluster2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],color="green",s=100,label="cluster3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],color="orange",s=100,label="cluster4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],color="blue",s=100,label="cluster5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="yellow",s=300,label="centroid")
plt.title("k means clustering")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()

