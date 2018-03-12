
# coding: utf-8

# In[1]:


print("Hello")


# In[6]:


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
arr_train_labels=np.array(mnist.train.labels)
arr_train_images=np.array(mnist.train.images)
arr_test_labels=np.array(mnist.validation.labels)
arr_test_images=np.array(mnist.validation.images)
print(len(arr_train_labels))
print(len(arr_train_images))
print(len(arr_test_labels))
print(len(arr_test_images))
list_x=[]
list_y=[]
c=0
print("training set")
# 4 is +1 and 9 is -1
for x in arr_train_labels:
    if(x[4]==1):
        list_x.append(arr_train_images[c])
        list_y.append(1)
    elif(x[9]==1):
        list_x.append(arr_train_images[c])
        list_y.append(-1)
    c=c+1
arr_train_x=np.array(list_x)
arr_train_y=np.array(list_y)
print(len(arr_train_x[10]))
print(len(arr_train_x))
print(len(arr_train_y))
print("test set")
c=0
list_x=[]
list_y=[]
for x in arr_test_labels:
    if(x[4]==1):
        list_x.append(arr_test_images[c])
        list_y.append(1)
    elif(x[9]==1):
        list_x.append(arr_test_images[c])
        list_y.append(-1)
    c=c+1
arr_test_x=np.array(list_x)
arr_test_y=np.array(list_y)
print(len(arr_test_x))
print(len(arr_test_y))


# In[24]:


def perceptron(epochs):
    weights = np.array(np.zeros(shape=(1,784)))
    length=len(arr_train_x)
    c=0
    e=0
    while e<epochs:
        flag=1
        #print(e)
        i=0
        while i<length:
            c=c+1
            #print(c)
            #print np.vdot(trainset[0],weights[0])
            if arr_train_y[i]*(np.vdot(arr_train_x[i],weights))<=0 :
                flag=0
                weights=weights+np.array((arr_train_y[i]*arr_train_x[i]))
            i=i+1
        #print(weights)
        if flag==1:
            #print("HERE")
            break
        e=e+1
    print(weights)
    print("End of run")


# In[25]:


perceptron(100)


# In[84]:


import math
def winnow(epochs):
    wp=np.full((1,784),1/(2*784))
    wn=np.full((1,784),1/(2*784))
    length=len(arr_train_x)
    e=0
    #print(len(arr_train_y[0]))
    while(e<epochs):
        i=0
        while(i<length):
            if((np.vdot(wp,arr_train_x[i])-np.vdot(wn,arr_train_x[i])))<=0:
                #wp=np.multiply(np.array(math.pow(math.e,0.1*np.vdot(arr_train_y[i],arr_train_x[i]))),np.array(wp))
                if(arr_train_y[i]<=0):
                    wp=wp*math.pow(math.e,0.1*(-1*np.array(arr_train_x[i])))
                    wn=wn*math.pow(math.e,-0.1*np.vdot(arr_train_y[i],arr_train_x[i]))
                else:
                    wp=wp*math.pow(math.e,np.dot(0.1,(arr_train_x[i])))
                j=0
                sum=0
                while j<784:
                    sum=sum+wp[j]+wn[j]
                    j=j+1
                wp=wp/sum
                wn=wn/sum
            i=i+1
        print(wp)
        print(wn)
        e=e+1
    print("End of run")


# In[85]:


winnow(1)


# In[1]:


import scipy.io as sio
import numpy as np

#Training 

mat_contents = sio.loadmat('mnist_all.mat')
#print len(mat_contents['train0'][0])
#let 0 be +1 and 1 be -1

trainset0 = mat_contents['train0']
trainset1 = mat_contents['train1']
weights = np.array(np.zeros(shape=(1,784)))

y0=1
y1=-1
pos=np.full((len(trainset0),1),y0)
neg=np.full((len(trainset1),1),y1)
directions=np.concatenate((np.array(pos),np.array(neg)),axis=0)

#Appending both train0 and train1 data together
trainset = np.concatenate((np.array(trainset0),np.array(trainset1)),axis=0)
print("hello")
print(len(trainset[0]))
length = len(trainset)
print(length) 

i=0
c=0

while i<length:
     c=c+1
     print(c)
     #print np.vdot(trainset[0],weights[0])
     if directions[i]*(np.vdot(np.array(trainset[i]),np.array(weights)))<=0 :
         weights=weights+np.array((directions[i]*trainset[i]))
         i=0
     else:
         i=i+1

print("WEIGHT:")
print (weights)
        
     #elif y1*(np.vdot(trainset1,weights)<=0):
     #weights[i]=weights[i]+(y1*trainset1)
            
#trainset1 = mat_contents['train1'][i]
     #print len(trainset0)

print (len(mat_contents['train0']))
print (len(mat_contents['train1']))

#print np.vdot(trainset0,weights)

#Testing

test_data0 = mat_contents['test0']
test_data1 = mat_contents['test1']
#print test_data1

#test_data = np.concatenate((np.array(test_data0),np.array(test_data1)),axis=0)
mistakes=0
print("MISTAKES:")
i=0
while i<len(test_data0):
    if (np.vdot(np.array(test_data0[i]),np.array(weights)))<=0:
        mistakes=mistakes+1
    i=i+1
print(mistakes)
i=0
while i<len(test_data1):
    if (np.vdot(np.array(test_data1[i]),np.array(weights)))>=0:
        mistakes=mistakes+1
    i=i+1
print(mistakes)

