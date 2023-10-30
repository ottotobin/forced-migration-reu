#!/usr/bin/env python
# coding: utf-8

# # Part 1 Preprocessing

# In[1]:


import numpy as np
import pandas as pd
import spacy
import re
from spacy.language import Language
import time
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer


# ### 1.1 Load the TASS training dataset

# In[2]:


tass = pd.read_csv('TASS_train_tweet.tsv', sep='\t', header=None)
tass.columns = ["No", "text", "label"]
tass = tass[tass["label"] != "NEU"]
eq = tass["label"] == "N"
tass = tass[["text","label"]]
tass["label"] = np.where(eq, 0, 1)
tass = tass.reset_index()
print(tass.shape)
tass.head()


# ### 1.2 Load the test dataset

# In[3]:


test_df = pd.read_csv('dev_es.tsv', sep='\t', header=None)
test_df.columns = ["No", "text", "label"]
test_df = test_df[["text", "label"]]
test_df = test_df[test_df["label"] != "NEU"]
test_df = test_df.reset_index()
test_df = test_df[["text","label"]]
test_df = test_df.reset_index()
eq = test_df["label"] == "N"
test_df["label"] = np.where(eq, 0, 1)
test_df.head()


# ### 1.5 unigram

# In[4]:


def text_cleaning(df):   
   
   spanish_tweets = np.array(df["text"])

   # !python -m spacy download es_core_news_sm
   pipeline = spacy.load("es_core_news_sm")
   
   # http://emailregex.com/
   email_re = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

   # replace = [ (pattern-to-replace, replacement),  ...]
   replace = [
       (r"<a[^>]*>(.*?)</a>", r"\1"),  # Matches most URLs
       (email_re, "email"),            # Matches emails
       (r"(?<=\d),(?=\d)", ""),        # Remove commas in numbers
       (r"\d+", "number"),              # Map digits to special token <numbr>
       (r"[\t\n\r\*\.\@\,\-\/]", " "), # Punctuation and other junk
       (r"\s+", " ")                   # Stips extra whitespace
   ]


   tweet = []
   for i, d in enumerate(spanish_tweets):
       for repl in replace:
           d = re.sub(repl[0], repl[1], d)
       tweet.append(d)


   # remove stop words and puctuations
   #@Language.component("fm")
   '''def ng20_preprocess(doc):
       tokens = [token for token in doc 
                 if not any((token.is_stop, token.is_punct))]
       tokens = [token.lemma_.lower().strip() for token in tokens]
       tokens = [token for token in tokens if token[:4]!="http" and token[:1]!="@" and token!="rt"]
       return " ".join(tokens)

   pipeline.add_pipe("fm")'''
   
   # pass text data through the pipeline
   sentences = []
   for t in spanish_tweets:
        doc = pipeline(t)
        filtered_words = [token.text for token in doc if not any((token.is_stop, token.is_punct))]
        filtered_text = " ".join(filtered_words)
        tokens = doc.text.split()
        sentences.append(filtered_text)
        '''txt = pipeline(t)
        txt = txt.split(sep=" ")
        sentences.append(txt) '''
    
   return sentences


# In[5]:


df1 = tass[["text","label"]]
df3 = test_df[["text","label"]]
full_df = pd.concat([df1, df3])
full_df.shape


# In[6]:


clean_full_text = text_cleaning(full_df)


# In[7]:


vectorizer = TfidfVectorizer(min_df=0.0001, max_df=0.95,
                                 analyzer='word', lowercase=False,ngram_range=(1, 1))
full_vector = vectorizer.fit_transform(clean_full_text)
#vectorizer.get_feature_names_out()


# In[8]:


full_vector_label = np.column_stack((full_vector.toarray(),np.array(full_df["label"])))
full_vector_label = pd.DataFrame(full_vector_label)
full_vector_label.head()


# In[9]:


tass_train, tass_test = train_test_split(full_vector_label, test_size=0.2, random_state=0)


# In[10]:


pos_train = tass_train[tass_train[3679]==1]
neg_train = tass_train[tass_train[3679]==0].sample(n=len(pos_train[3679]), random_state=0)
tass_train = pd.concat([pos_train, neg_train], axis=0).sample(frac=1)

pos_test = tass_test[tass_test[3679]==1]
neg_test = tass_test[tass_test[3679]==0].sample(n=len(pos_test[3679]), random_state=0)
tass_test = pd.concat([pos_test, neg_test], axis=0).sample(frac=1)


# In[11]:


X_train = tass_train.iloc[:,:3679]
y_train = tass_train.iloc[:,3679]
X_test = tass_test.iloc[:,:3679]
y_test = tass_test.iloc[:,3679]


# In[12]:


X_test.shape
X_train


# In[13]:


X_train.shape
sum(y_test)/len(y_test)


# # Part 2 Machine learning models

# 2.1 K-fold

# In[14]:


from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
#
# Create an instance of Pipeline
#
# pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
#
# Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
#
# strtfdKFold = StratifiedKFold(n_splits=5)
# kfold = strtfdKFold.split(X_train, y_train)


# ### 2.1 KNN

# In[15]:


classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)


# In[16]:


# y_pred = classifier.predict(X_val)
# cm = confusion_matrix(y_val, y_pred)
# print("confusion matrix on the validation dataset is: \n", cm)
# ac = accuracy_score(y_val,y_pred)
# print("accuracy is:", ac)


# In[17]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix on the test dataset is: \n", cm)
ac = accuracy_score(y_test,y_pred)
print("accuracy is:", ac)
f1 = f1_score(y_test,y_pred)
print("f1 is:", f1)


# In[18]:


# results=cross_val_score(classifier,X_train,y_train,cv=kfold)
# print(results)
# print(np.mean(results))


# ### 2.2 Naive Bayes

# In[19]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[20]:


# y_pred = classifier.predict(X_val.toarray())
# cm = confusion_matrix(y_val, y_pred)
# print("confusion matrixon the validation dataset is: \n", cm)
# ac = accuracy_score(y_val,y_pred)
# print("accuracy is:", ac)


# In[21]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix on the test dataset is: \n", cm)
ac = accuracy_score(y_test,y_pred)
print("accuracy is:", ac)
f1 = f1_score(y_test,y_pred)
print("f1 is:", f1)


# ### 2.3 Decision Tree

# In[22]:


classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[23]:


# y_pred = classifier.predict(X_val)
# cm = confusion_matrix(y_val, y_pred)
# print("confusion matrix on the validation dataset is: \n", cm)
# ac = accuracy_score(y_val,y_pred)
# print("accuracy is:", ac)


# In[24]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix on the test dataset is: \n", cm)
ac = accuracy_score(y_test,y_pred)
print("accuracy is:", ac)
f1 = f1_score(y_test,y_pred)
print("f1 is:", f1)


# In[25]:


# results=cross_val_score(classifier,X_train,y_train,cv=kfold)
# print(results)
# print(np.mean(results))


# ### 2.4 Random forest

# In[26]:


classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)


# In[27]:


# y_pred = classifier.predict(X_val)
# cm = confusion_matrix(y_val, y_pred)
# print("confusion matrix on the validation dataset is: \n", cm)
# ac = accuracy_score(y_val,y_pred)
# print("accuracy is:", ac)


# In[28]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix on the test dataset is: \n", cm)
ac = accuracy_score(y_test,y_pred)
print("accuracy is:", ac)
f1 = f1_score(y_test,y_pred)
print("f1 is:", f1)


# ### 2.5 SVM

# In[29]:


classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
classifier.fit(X_train, y_train)


# In[30]:


# y_pred = classifier.predict(X_val.toarray())
# cm = confusion_matrix(y_val, y_pred)
# print("confusion matrix on the validation dataset is: \n", cm)
# ac = accuracy_score(y_val,y_pred)
# print("accuracy is:", ac)


# In[31]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix on the test dataset is: \n", cm)
ac = accuracy_score(y_test,y_pred)
print("accuracy is:", ac)
f1 = f1_score(y_test,y_pred)
print("f1 is:", f1)


# # Part 3 Visualization

# ### 3.1 PCA

# In[32]:


pca = PCA(n_components=2)
train_pca_result = pca.fit_transform(train_embedding)
print(train_pca_result.shape)
train_pca_df = pd.DataFrame(train_pca_result)
train_pca_df.columns = ["pca_1", "pca_2"]
train_pca_df["label"]= tass["label"]
train_pca_df.head()


# In[ ]:


val_pca_result = pca.fit_transform(val_embedding)
print(val_pca_result.shape)
val_pca_df = pd.DataFrame(val_pca_result)
val_pca_df.columns = ["pca_1", "pca_2"]
val_pca_df["label"]= val_df["label"]
val_pca_df.head()


# In[ ]:


test_pca_result = pca.fit_transform(test_embedding)
print(val_pca_result.shape)
test_pca_df = pd.DataFrame(test_pca_result)
test_pca_df.columns = ["pca_1", "pca_2"]
test_pca_df["label"]= test_df["label"]
test_pca_df.head()


# In[ ]:


plt.figure(figsize=(6,4))
sns.scatterplot(
    x="pca_1", y="pca_2",
    hue="label",
    palette=sns.color_palette("pastel",2),
    data=train_pca_df,
    legend="full",
    #alpha=0.3,
    s = 24
)
plt.title("PCA scatter plot of training dataset")


# In[ ]:


plt.figure(figsize=(6,4))
sns.scatterplot(
    x="pca_1", y="pca_2",
    hue="label",
    palette=sns.color_palette("pastel",2),
    data=val_pca_df,
    legend="full",
    #alpha=0.3,
    s = 24
)
plt.title("PCA scatter plot of validation dataset")


# In[ ]:


plt.figure(figsize=(6,4))
sns.scatterplot(
    x="pca_1", y="pca_2",
    hue="label",
    palette=sns.color_palette("pastel",2),
    data=test_pca_df,
    legend="full",
    #alpha=0.3,
    s = 24
)
plt.title("PCA scatter plot of testing dataset")


# There is a clear pattern in these three scatter plots. After we set two principal componets as x and y axis, the scatters with different labels(Negative or Positve) gathered into two clusters. Scatters labeled as Negative showed up more often in the bottom half and skewed to the right, and scatters labeled as Positive were mostly in the top and skewed to the left. Overall, this proves that the result of word embedding (vecterization) contains the information from the original text data. In the future, we might be able to apply classification model basing on the result of PCA to reduce computaional cost since it is one of the effective ways to reduce dimentionality of the dataset.

# ### 3.2 t-SNE

# In[ ]:


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
train_tsne_results = tsne.fit_transform(train_embedding)
train_tsen_df = pd.DataFrame(train_tsne_results)
train_tsen_df.columns = ["tsne_1", "tsne_2"]
train_tsen_df["label"] = tass["label"]
train_tsen_df.head()


# In[ ]:


val_tsne_results = tsne.fit_transform(val_embedding)
val_tsen_df = pd.DataFrame(val_tsne_results)
val_tsen_df.columns = ["tsne_1", "tsne_2"]
val_tsen_df["label"] = val_df["label"]
val_tsen_df.head()


# In[ ]:


test_tsne_results = tsne.fit_transform(test_embedding)
test_tsen_df = pd.DataFrame(test_tsne_results)
test_tsen_df.columns = ["tsne_1", "tsne_2"]
test_tsen_df["label"] = test_df["label"]
test_tsen_df.head()


# In[ ]:


fig, ax = plt.subplots()
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="label",
    palette=sns.color_palette("hls", 2),
    data=train_tsen_df,
    legend="full",
    alpha=0.7,
    ax=ax
)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 30)
plt.title("t-SNE scatter plot of training dataset")
plt.show()


# In[ ]:


fig, ax = plt.subplots()
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="label",
    palette=sns.color_palette("hls", 2),
    data=val_tsen_df,
    legend="full",
    alpha=0.7,
    ax=ax
)
ax.set_xlim(-50, 30)
ax.set_ylim(-30, 40)


# In[ ]:


###### plt.figure(figsize=(6,4))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="label",
    palette=sns.color_palette("hls", 2),
    data=test_tsen_df,
    legend="full",
    alpha=0.7
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




