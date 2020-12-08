#!/usr/bin/env python
# coding: utf-8

# # Classify Song Genres from Audio Data
# ### Final Project

# ## Task-1: Preparing our Data Set

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


track = pd.read_csv('fma-rock-vs-hiphop.csv')
echonest_metrics = pd.read_json('echonest-metrics.json',precise_float = True)


# In[3]:


track.head(2)


# In[4]:


echonest_metrics.head()


# In[5]:


echo_tracks = pd.merge(left = track[['track_id', 'genre_top']], right=echonest_metrics, on='track_id')


# In[6]:


echo_tracks.head()


# In[7]:


echo_tracks.describe()


# In[8]:


echo_tracks.info()


# ## Task-2: Pairwise relationships between continuous variables

# In[9]:


corr_matrix = echo_tracks.corr(method='pearson')
corr_matrix.style.background_gradient()


# ## Task-3: Normalizing the feature data

# In[10]:


# Define our features 
features = echo_tracks.drop(['genre_top', 'track_id'], axis=1)


# In[11]:


# Define our labels
labels = echo_tracks['genre_top']


# In[12]:


# Import the StandardScaler
from sklearn.preprocessing import StandardScaler


# In[13]:


# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
pd.DataFrame(scaled_train_features).head()


# ## Task-4: Principal Component Analysis on our scaled data

# In[14]:


# This is just to make plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# Import our plotting module, and PCA class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[16]:


# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_


# In[17]:


# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(8), exp_variance)
ax.set_xlabel('Principal Component #')


# ## Task-5: Further Visualization of PCA

# ### The above Scree plot does not gave us a clear Elbow, So we are going with Cumulative Explained Variance Plot

# In[18]:


# Get our explained variance ratios from PCA using all features
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(scaled_train_features)


# In[19]:


exp_variance=pca.explained_variance_ratio_
cum_exp_variance=np.cumsum(exp_variance)
cum_exp_variance


# In[20]:


# Plot the cumulative explained variance and draw a dashed line at 0.95.
fig,ax=plt.subplots()
ax.plot(cum_exp_variance,color='r')
ax.axhline(y=0.95,linestyle='--')
plt.grid()
plt.show()


# ### From the above plot we can see optimal number of features needed is 6, so we assign 6 for n_components for PCA

# In[21]:


n_components = 6

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)


# In[22]:


pca_projection.shape


# ## Task-6: Train a Decision Tree to Classify Genre

# In[23]:


# Splitting data to Train and Test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(pca_projection,labels,test_size=0.3)


# In[24]:


# Model Building
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(features_train,labels_train)
dtc_pred=dtc.predict(features_test)


# In[25]:


# Calculating Accuracy Score of the Model
from sklearn.metrics import accuracy_score
score=accuracy_score(labels_test,dtc_pred)
score


# In[26]:


# Comparing both original and predicted  labels for valuation
output=dtc.predict(pca_projection)
Genre=pd.DataFrame({"Genre Actual":labels,"Genre Predicted":output})
Genre.head()


# ### Although our tree's performance is decent, it's a bad idea to immediately assume that, its therefore the perfect tool for this job -- there's always the possibility of other models that will perform even better! It's always a worthwhile idea to at least test a few other algorithms and find the one that's best for our data.
# 

# ## Task-7: Building a Logistic Model and  Comparing it with Decision Tree

# In[27]:


# Importing a logistic regression
from sklearn.linear_model import LogisticRegression


# In[28]:


# Training the logistic model 
lr=LogisticRegression()
lr.fit(features_train,labels_train) 
lr_pred=lr.predict(features_test)


# In[29]:


from sklearn.metrics import accuracy_score
score=accuracy_score(labels_test,lr_pred)
score


# In[30]:


# Importing a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(features_train,labels_train)
dtc_pred=dtc.predict(features_test)


# In[31]:


from sklearn.metrics import accuracy_score
score=accuracy_score(labels_test,dtc_pred)
score


# ### Thus we can observe that the accuracy score of the logistic model (88.2%) is little  higher than that of decision tree (85.3%)

# In[32]:


from sklearn.metrics import classification_report
report_tree=classification_report(labels_test,dtc_pred)
report_logit=classification_report(labels_test,lr_pred)
print(" DECISION TREE CLASSIFICATION REPORT:\n", report_tree)
print(" LOGISTIC MODEL CLASSIFICATION REPORT:\n", report_logit)


# ## Task-8: Balance our Data for Greater Performance

# ### Both our models do similarly well, boasting an average precision of 87% each. However, looking at our classification report, we can see that rock songs are fairly well classified, but hip-hop songs are disproportionately misclassified as rock songs.

# In[33]:


# Subsetting the hip-hop tracks and then subsetting the rock tracks.
hhop_tracks=echo_tracks.loc[echo_tracks['genre_top']== 'Hip-Hop']
rock_tracks=echo_tracks.loc[echo_tracks['genre_top']== 'Rock']


# In[34]:


# Balancing the no of rock song with the no of hip-hop songs
rock_tracks=rock_tracks.sample(n=len(hhop_tracks))


# In[35]:


# Now we are merging the dataframe of hhop_track and rock_tracks
final_rock_hop=pd.concat([hhop_tracks,rock_tracks])


# In[36]:


# The features ,labels and pca projection are created for the final balanced dataframe
features = final_rock_hop.drop(['genre_top', 'track_id'], axis=1) 
labels = final_rock_hop['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))


# In[37]:


# Splitting the final balanced data into test and train with the pca_projection.
features_train,features_test,labels_train,labels_test= train_test_split(pca_projection,labels)


# ## Task-9 : Does balancing our dataset improve model bias?

# In[38]:


# Train our decision tree on the balanced data
dtc = DecisionTreeClassifier(random_state=10)
dtc.fit(features_train,labels_train)
pred_labels_tree = dtc.predict(features_test)

# Train our logistic regression and predict labels for the test set
lr = LogisticRegression(random_state=10)
lr.fit(features_train, labels_train)
pred_labels_logit = lr.predict(features_test)

# Create the classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(labels_test, pred_labels_tree)
class_rep_log = classification_report(labels_test, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)


# ### Thus we can see that balancing our data has removed bias towards the more prevalent class. To get a good sense of how well our models are actually performing, we can apply what’s called cross-validation(CV). This step allows us to compare models in a more rigorous fashion.

# ## Task-10: Using cross-validation to evaluate our models
# 
# ### We will use what’s known as K-fold cross-validation here. K-fold first splits the data into K different, equally sized subsets. Then, it iteratively uses each subset as a test set while using the remainder of the data as train sets.

# In[39]:


from sklearn.model_selection import KFold, cross_val_score

# Set up our K-fold cross-validation
kf = KFold(n_splits=10, random_state=10)

tree = DecisionTreeClassifier(random_state=10)
logreg = LogisticRegression(random_state=10)

# Train our models using KFold cv
tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)
logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)

# Print the mean of each array of scores
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))


# ### Thus we can be say that our model will generalize 75% of the times on the future unseen data points.
