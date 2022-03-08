#!/usr/bin/env python
# coding: utf-8

# In[18]:


### week 1 Python data Sctructures
my_tuple = (120, 80, 30) # Can NOT be modified
animal_list = ['cat', 'dog', 'pp'] # Can be modified
my_dic = {'mahdi':27, 'eli': 30} # pairs of Key:Value elemnets , 

# Pandas data Structures: dataframes, Series(not important, it's like dictionary)
import pandas as pd
my_data = {'student ID':[98745081, 98762514],
           'name':['mahdi', 'nima'],
           'grade': [20, 4]
          }
my_dataframe = pd.DataFrame(my_data)


# In[13]:


my_tuple[1]
animal_list[1]
my_dic['mahdi']


# In[10]:


animal_list.append('bee')
animal_list.remove('pp')


# In[24]:


animal_list[2]
my_dataframe


# In[26]:


my_dataframe.name
my_dataframe['age'] = [27, 28.5]
my_dataframe


# In[ ]:


#2 Reading data
df = pd.read_csv('./readonly/physics_students.csv')

df.head()
df.tail()
df.columns
df.shape # rows and cols

df['degree'] # a column
df[5:10] # some observations
df[5:10]['degree']
df[df.age > 20]

df.describe() # summarizing statisically
df[age].mean()
df[age].sd()


# In[8]:


### WEEK 2 - Image Analysis (+ a little Genome Sequencing)
#https://www.coursera.org/learn/datascimed/ungradedLab/B4Iyk/image-analysis-programming-task/lab?path=%2Fnotebooks%2FWK2_Image_Analysis_Task.ipynb

import os
import pydicom
import SimpleITK
import numpy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')


# In[6]:


## Part 2: Loading the data
PathDicom = "./readonly/MyHead/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file is DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
            
lstFilesDCM[:5]


# In[7]:


# Reading a DICOM file
HeadDs = pydicom.read_file(lstFilesDCM[0])

# Getting metadata
HeadDs.PatientPosition # HFS stands for Head First-Supine. This means that the patient’s head was positioned toward the front of the imaging equipment and it was in an upward direction.

HeadDs.StudyDate # to get the date the study started, YYYYMMDD.

HeadDs.Modality # to get the image modality, MR(MRI) | CT(CT) | PET(PT)


# In[ ]:


## Part 3: Visualisation
# Preparing for visualisation, In order to plot the data with Matplotlib, FIRST: combine the pixel data from all DICOM files (i.e. from all slices) into a 3D dataset
CalcPixelDims = (int(HeadDs.Rows), int(HeadDs.Columns), len(lstFilesDCM))
CalcPixelDims

HeadImgArray = numpy.zeros(CalcPixelDims, dtype=HeadDs.pixel_array.dtype)

for filenameDCM in lstFilesDCM:
    ds = pydicom.read_file(filenameDCM)
    HeadImgArray[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    
# SECOND : specify appropriate coordinate axes
CalcPixelSpacing = (float(HeadDs.PixelSpacing[0]), float(HeadDs.PixelSpacing[1]), float(HeadDs.SliceThickness))

x = numpy.arange(0.0, (CalcPixelDims[0]+1)*CalcPixelSpacing[0], CalcPixelSpacing[0])
y = numpy.arange(0.0, (CalcPixelDims[1]+1)*CalcPixelSpacing[1], CalcPixelSpacing[1])
z = numpy.arange(0.0, (CalcPixelDims[2]+1)*CalcPixelSpacing[2], CalcPixelSpacing[2])

# visulizing
plt.figure(dpi=300)
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, numpy.flipud(HeadImgArray[:, :, 125]))


# In[ ]:


### Specifying a helper function, that quickly plots a 2D SimpleITK image with a greyscale colourmap and accompanying axes.
def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


# In[ ]:


### Loading the data in SimpleITK
reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenamesDICOM)
img3DOriginal = reader.Execute()

# for simplicity, we'll segment a 2D slice of the 3D image (rather than the entire 3D image)
imgOriginal = img3DOriginal[:,:,50] 

### Visualising the original data
sitk_show(imgOriginal)


# In[ ]:


### Smoothing,  reducing noise within an image or producing a less pixelated image
imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                    timeStep=0.125,
                                    numberOfIterations=5)

sitk_show(imgSmooth)


# In[ ]:


### Segmentation with the ConnectedThreshold filter
lstSeeds = [(150,75)] # the starting point, which we know is e.g. white matter.

imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                              seedList=lstSeeds, 
                                              lower=130, 
                                              upper=190,
                                              replaceValue=1)

# preprocess, overlay, and visualizing the result
imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatter))


# In[ ]:


#### Hole-filling of the segmented white matter
imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                          radius=[2]*3,
                                                          majorityThreshold=1,
                                                          backgroundValue=0,
                                                          foregroundValue=1)

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))


# In[ ]:


#### Segmentation and hole-filling of grey matter
# we just repeat the whole above process for grey matter parts...
lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43)]

imgGreyMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                             seedList=lstSeeds, 
                                             lower=150, 
                                             upper=270,
                                             replaceValue=2)

imgGreyMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGreyMatter,
                                                         radius=[2]*3,
                                                         majorityThreshold=1,
                                                         backgroundValue=0,
                                                         foregroundValue=2) # labelGrayMatter

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgGreyMatterNoHoles))


# In[ ]:


#### Combining the white and grey matter (combining the 2 label fields)
imgLabels = imgWhiteMatterNoHoles | imgGreyMatterNoHoles

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))

imgMask = (imgWhiteMatterNoHoles/1) * (imgGreyMatterNoHoles/2)
imgMask2 = SimpleITK.Cast(imgMask, imgWhiteMatterNoHoles.GetPixelIDValue())
imgWhiteMatterNoHoles = imgWhiteMatterNoHoles - (imgMask2*1)
imgLabels2 = imgWhiteMatterNoHoles + imgGreyMatterNoHoles

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels2))


# In[20]:


### week 3 - Machine Learning
import sklearn

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("key of iris dataset ARE: \n", iris_dataset.keys()) # inspecting data
print(iris_dataset['DESCR'][:200] + "\n.......")
print("Feature names ARE: \n", iris_dataset['feature_names'])
print("Target names ARE: ", iris_dataset['target_names']) # target_names = the class labels
print("Shape of target IS: ", iris_dataset['target'].shape)
print("First two elements in target ARE: ", iris_dataset['target'][:2])

print("The shape of data IS: ", iris_dataset['data'].shape)
print("First three rows of data ARE:\n", iris_dataset['data'][:3])


# In[25]:


# Splitting our dataset into training  and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


# In[32]:


# Our first model: KNN - K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

# Evaluating the model 
print("Test set score: ", knn.score(X_test, y_test)) 
print("Test set score rounded to three decimal places: {:.3f}".format(knn.score(X_test, y_test)))

# Make Predictions
import numpy as np
X_unseen = np.array([[5.3, 2.7, 1, 0.3]])

prediction = knn.predict(X_unseen)

print("\n Prediction label: ", prediction)
print("\n Predicted target name: ", iris_dataset['target_names'][prediction])


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=7)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=12)
tree.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# In[35]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=12)
tree.fit(X_train, y_train)

print("Accuracy on training set: ", tree.score(X_train, y_train))
print("Accuracy on test set: ", tree.score(X_test, y_test))


# In[36]:


prediction = tree.predict(X_unseen)

print("Prediction label: ", prediction)
print("Predicted target name: ", iris_dataset['target_names'][prediction])


# In[40]:


### WEEK 3 - Machine Learning; Programming Assignmet 
#https://www.coursera.org/learn/datascimed/ungradedLab/Ae3kz/programming-assignment-notebook/lab?path=%2Fnotebooks%2FProgramming_Assignment.ipynb
import pandas as pd
import sklearn
import numpy as np


# In[2]:


### WEEK 4 - NLP, Natural Language Processing
import nltk
import docx2txt
from nltk.corpus import stopwords


# In[4]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[8]:


## Processing a Biopsy Report
# Loading the data
text = docx2txt.process('./readonly/Biopsy_Report.docx') #using the process method from the docx2txt Python package to convert from .docx into Plain_Text.
type(text)
text[:160] # to get the first 160 characters in text


# In[ ]:


## Tokenization
tokens = nltk.word_tokenize(text) # into Words !
tokens[:10] # to get the first 10 elements of tokens

# Cleaning, (i.e. removing the ANDs, ORs, AREs, etc...)
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

print("Number of tokens including stop words:  ",len(tokens))
print("Number of tokens excluding stop words:  ",len(clean_tokens))

# Frequency Distribution of some certain Words
freq = nltk.FreqDist(tokens)
freq.most_common(10) # the most commons

print("Frequency of lesion:  ", freq["lesion"]) # freq of "lesion"
print("Frequency of lesions: ", freq["lesions"])
print("Frequency of LESION:  ", freq["LESION"])
print("Frequency of LESIONS: ", freq["LESIONS"])

# Lower case vs. upper case text
lowercase_tokens = [t.lower() for t in tokens]
lowercase_tokens[:10]

lowercase_freq = nltk.FreqDist(lowercase_tokens)
print("Frequency of lesion:  ", lowercase_freq["lesion"]) # freq of "lesion"
print("Frequency of lesions: ", lowercase_freq["lesions"])
print("Frequency of LESION:  ", lowercase_freq["LESION"])
print("Frequency of LESIONS: ", lowercase_freq["LESIONS"])

# Stemming, the process of reducing a word to its stem.
stemmer = nltk.PorterStemmer()
stem_tokens = lowercase_tokens
stem_tokens[:] = [stemmer.stem(lt) for lt in lowercase_tokens]

stem_freq = nltk.FreqDist(stem_tokens)
print("Frequency of lesion:  ", stem_freq["lesion"]) # freq of "lesion"
print("Frequency of lesions: ", stem_freq["lesions"])
print("Frequency of LESION:  ", stem_freq["LESION"])
print("Frequency of LESIONS: ", stem_freq["LESIONS"])


# In[9]:


## Processing a Medical Note
# Load the Data
content = docx2txt.process('./readonly/Medical_Note.docx')
content[:160]

# Tokenisation
sents = nltk.sent_tokenize(content) # into Sentences !
sents[:4]

# now, just the Sent[1] for further processing
medical_tokens = nltk.word_tokenize(sents[1])
medical_tokens


# In[ ]:


## Part-of-Speech Tagging, processes a sequence of words and attaches a part of speech tag to each word
# meaning that, we'll know which part is Adjective OR Noun OR Verb, etc ...
tagged = nltk.pos_tag(medical_tokens)
tagged


# In[11]:


## Named Entity Recognition, finding entities in text & classifying them as persons, locations, date,...
entities = nltk.ne_chunk(tagged)
print(entities)


# In[ ]:




