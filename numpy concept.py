#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
a=np.array([1,2,3,4,5])


# In[31]:


a.dtype


# In[3]:


print(a)


# In[5]:


np.array([1,2,3,4,5,6.0])


# In[6]:


a=np.array([[1,3],[3,4]])


# In[9]:


np.ndim(a)


# In[14]:


np.array([1,2,3], ndmin=2)


# In[18]:


np.array([1,2,3], ndmin=2)


# In[20]:


c=np.array([[1,2,3],[4,5,6]])
print(np.ndim(c))


# In[23]:


d=np.array([1,2,3,4])
np.ndim(d)


# In[ ]:





# In[29]:


s=np.array([[[1,2,3],[4,5,7]],[[3,4,5],[6,7,9]]])
np.ndim(s)


# In[30]:


np.array([1,2,3] , dtype=str)


# In[32]:


a=np.array([1,2,3])
a[0]


# In[33]:


a[1]


# In[34]:


a[2]


# In[35]:


b=np.array([[1,2,3],[4,5,6]])
b[0,0]


# In[37]:


b[0,1]


# In[38]:


b[0,2]


# In[39]:


b[1,0]


# In[45]:


c=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[3,4,5]]])
print(c)
c[0,0]


# In[41]:


c[0,0,0]


# In[42]:


c[0,0,1]


# In[43]:


c[0,0,2]


# In[44]:


c[0,1,0]


# In[46]:


c[1,0,0]


# In[47]:


c[1,1,0]


# In[48]:


c[1][1][0]


# In[53]:


l=np.array([1,2,3,4,5,6])
np.where(5)


# In[54]:


issubclass(np.matrix,np.ndarray)


# In[56]:


a=np.array([1,2,3,4])


# In[60]:


a[0]=67


# In[61]:


a


# In[62]:


b=np.copy(a)


# In[63]:


b


# In[64]:


n=([1,2,3,4])


# In[65]:


m=n


# In[66]:


o.copy(m)


# In[69]:


m[0]=6
m


# In[ ]:


a=arr
b=np.copy(a)


# In[70]:


a[0]=6


# In[71]:


a


# In[72]:


h=np.array([1,2,3,4])


# In[74]:


m=h


# In[75]:


i=np.copy(m)


# In[77]:


m[0]=9


# In[78]:


m


# In[79]:


i


# In[80]:


a=([1,2,3,4])
b=np.copy(a)


# In[82]:


a[0]=5


# In[83]:


a


# In[84]:


b


# In[2]:


import numpy as np


# In[4]:


arr=np.array([1,2,3,4])
arr[-1]
 


# In[5]:


for i in arr:
    print(i)


# In[6]:


np.fromfunction(lambda i,j : i==j,(4,4))


# In[7]:


np.fromfunction(lambda i,j : i*j,(2,2) )


# In[8]:


np.fromfunction(lambda i,j : i**j,(2,2))


# In[11]:


(rangefor i in range(5):
  print (i*i)


# In[14]:


np.fromiter(range(5),dtype= int)


# In[22]:


np.fromstring('1,2,3,4' , sep= ',',dtype=int)


# In[26]:


l=[3,4,5,6,7,8]
np.array(l)


# In[28]:


arr1=np.array([1,2,3,4])


# In[29]:


arr1.ndim


# In[30]:


arr1.size


# In[31]:


arr1.shape


# In[33]:


arr2=np.array([[[1,2,3],[4,5,6]],[[4,5,6],[7,8,9]]])


# In[34]:


arr2.ndim


# In[35]:


arr2.size


# In[36]:


arr2.shape


# In[37]:


arr2


# In[39]:


a=np.array([1,2,3,4] , ndmin=3)


# In[43]:


arr3=np.array([1,2,3,4], ndmin=5)


# In[44]:


arr3.shape


# In[2]:


import numpy as np
rr2=np.array([[[[1,2,3],[4,5,6]],[[4,5,6],[7,8,9]],[[1,2,3],[4,5,6]]]])


# In[3]:


rr2.ndim


# In[4]:


rr2.size

rr2
# In[6]:


rr2


# In[7]:


rr2.shape


# In[11]:


list(range(5))


# In[13]:


np.arange(6.6)


# In[18]:


np.arange(6.6,9,0.3)


# In[19]:


np.arange(1,9)


# In[20]:


np.linspace(2,3)


# In[25]:


np.zeros([2,3])


# In[29]:


np.ones([2,2,4,5])


# 

# In[30]:


np.ones([1,2])+5


# In[32]:


np.empty(5)


# In[33]:


np.eye(3)


# In[37]:


np.linspace(2,4,4,endpoint=False,dtype=int)


# In[38]:


np.logspace(2,3,5)


# In[42]:


np.logspace([2],[3])


# In[43]:


np.logspace(3)


# In[44]:


d=np.linspace(4,7,40)


# In[45]:


d


# In[46]:


d.ndim


# In[48]:


d.reshape(40,1)


# In[49]:


d.ndim


# In[50]:


d.reshape(1,2,20)


# In[51]:


d.ndim


# In[54]:


np.array([2,2])


# In[55]:


np.random.rand(3)


# In[56]:


np.random.randn(5)


# In[7]:


import numpy as np
np.random.randn(2,3)


# In[8]:


np.random.randint(5)


# In[9]:


np.random.randint(2,5)


# In[1]:


import numpy as np


# In[4]:


np.random.randn(2,3)


# In[5]:


np.random.randint(5)


# In[6]:


np.random.randint(5,60,(4,4))


# In[7]:


arr=np.random.randint(3,9,(3,3))


# In[8]:


arr


# In[9]:


arr.reshape(1,9)


# In[10]:


arr.reshape(9,1)


# In[11]:


arr.reshape(3,3)


# In[12]:


arr.reshape(1,1,9)


# In[13]:


arr.reshape(3,-1)


# In[14]:


arr.reshape(1,-1)


# In[15]:


arr.max()


# In[16]:


arr.min()


# In[17]:


arr.reshape(1,-2)


# In[18]:


arr=np.random.randint(4,100arrr,(5,5))


# In[19]:


arr


# In[20]:


arr[3:,3:]


# In[21]:


arr[:,[1,3]]


# In[22]:


arr


# In[23]:


arr


# In[24]:


arr>10


# In[26]:


(arr[arr>10])


# In[27]:


arr


# In[28]:


arr*arr


# In[29]:


arr1=np.random.randint(2,4(3,3))


# In[31]:


arr1=np.random.randint(2,4,(3,3))


# In[32]:


arr1


# In[33]:


arr2=np.random.randint(2,4,(3,3))

arr2
# In[34]:


arr2


# In[35]:


arr1*arr2


# In[36]:


arr1@arr2


# In[37]:


arr/0


# In[40]:


arr=np.zeros([3,3])


# In[41]:


arr


# In[1]:


import numpy as np


# In[2]:


arr=([1,2,3,4])


# In[3]:


np.ndim(arr)


# In[4]:


arr.ndim


# In[5]:


arr.shape


# In[6]:


np.shape(arr)


# In[7]:


np.size(arr)


# In[8]:


arr[0]


# In[9]:


arr.max()


# In[10]:


np.arr(max)


# In[11]:


arr.min()


# In[12]:


arr


# In[13]:


arr.max()


# In[14]:


arr1=np.array([[1,2,3],[4,5,6]])


# In[15]:


arr1.max()


# In[16]:


arr2=np.array([1,2,3,4])
arr2.max()


# In[17]:


arr3=np.random.randint(5)


# In[18]:


arr3


# In[19]:


arr3.ndim


# In[20]:


np.ndim(arr3)


# In[21]:


arr4=np.random.randint(5,6,(4,4))


# In[22]:


arr4


# In[23]:


np.ndim(arr4)


# In[27]:


arr5=np.random.randint(6,7,(2,3,4))


# arr5

# In[28]:


arr5


# In[29]:


arr6=np.random.randint(5,6,(2,2))


# In[30]:


arr6


# In[36]:


arr6[;,[4]]


# In[37]:


arr6[0]


# In[38]:


arr6[0,0]


# In[43]:


np.arr6([0,0],[1,0])


# In[46]:


arr6([0,1],[1,0])


# In[47]:


arr7=np.random.randint(7,9,(3,3))


# In[48]:


arr7


# In[58]:


arr7[0:2,[0,2]]


# In[55]:


arr7[[0,1],[1,0]]


# In[63]:


arr7[0:2,[1:]]


# In[67]:


arr8=np.random.randint(4,9,(4,4))


# In[68]:


arr8


# In[70]:


arr8[2:,2:]


# In[72]:


arr8[1:3,[0,1]]


# In[73]:


arr8


# In[75]:


arr(1,0)


# In[81]:


arr9=np.random.randint(7,9,(2,3,4))


# In[ ]:





# In[82]:


arr9


# ## arr9[0,]
arr9[0,2,1]
# In[ ]:





# In[84]:


arr9[0,2,2]


# In[85]:


arr=np.array([1,2,3],ndmin=3)


# In[86]:


arr


# In[89]:


arr=np.zeros((3,4))


# In[90]:


arr


# In[91]:


a=np.array([1,2,3,4])
arr+a


# In[2]:


import numpy as np
arr=np.random.randint(3,8,(4,4))


# In[3]:


arr


# In[5]:


arr.T


# In[10]:


arr.reshape(1,8,2)


# In[11]:


np.sqrt(arr)


# In[12]:


np.exp(arr)


# In[13]:


np.log10(arr)


# In[14]:


np.log2(arr)


# In[21]:


np.random.seed(30)
np.random.rand(2,4)


# In[19]:


np.random.rand(2,4)


# In[22]:


a=np.array([1.1,2.1,3.1])


# In[24]:


a.dtype


# In[27]:


a=np.array([1.1],dtype=int)


# In[28]:


a.dtype


# In[29]:


new_a=a.astype('i')


# In[30]:


new_a


# In[31]:


new_b=a.astype('bool')


# In[32]:


new_b


# In[33]:


a1=a.copy


# In[34]:


a1


# In[35]:


a


# In[36]:


b=np.array([1,2,3,4])


# In[37]:


b1=b.copy()


# In[40]:


b


# In[42]:


b1


# In[43]:


print(b1)


# In[44]:


b1


# In[46]:


arr=np.array([1,2,3,4])


# In[49]:


arr1=arr.copy()


# In[50]:


arr1


# In[52]:


arr[1]=10


# In[53]:


arr


# In[54]:


arr1


# In[55]:


arr2=arr1.view


# In[56]:


arr2


# In[57]:


arr3=np.array([1,2,3,4,5])


# In[58]:


arr4=arr3.view()


# In[59]:


arr4


# In[60]:


arr4[1]=5


# In[62]:


arr3


# In[63]:


x=arr3


# In[64]:


x[1]=7


# In[65]:


arr3


# In[66]:


#creating range


# In[67]:


a=np.arange(10)


# In[68]:


a


# In[69]:


b=a.reshape(2,5)


# In[70]:


b


# In[71]:


for i in b:


# In[75]:


for element in b:
    print(element)


# In[77]:


for numbers in element:
    print(numbers)


# In[78]:


for row in b:
    for element in row:
        print(element)
   


# In[79]:


for i in np.nditer(b):
    print(i)


# In[82]:


arr=np.array([1,2,3])
for x,y in np.ndenumerate(arr):
  print(x,y)


# In[86]:


arr1=np.arange(1,10)
print(np.split(arr1,3))


# In[1]:


import numpy as np


# In[2]:


arr=np.array([[1,2],[2,2],[3,3]])


# In[3]:


arr


# In[4]:


np.insert(arr,1,5,axis=1)


# In[6]:


np.insert(arr,[1],[[1],[2],[3]],axis=1)


# In[7]:


np.append([1,2,3],[[4,5,6],[7,8,9]])


# In[8]:


np.append([[1,2,3],[4,5,6]],[[7,8,9]],axis=0)


# In[11]:


x=np.arange(1,10).reshape(3,3)


# In[18]:


np.delete(x,2,0)


# In[20]:


arr=np.random.rand()


# In[21]:


arr


# In[22]:


arr1=np.random.rand(3)


# In[23]:


arr1


# In[27]:



np.random.choice([1,2,3],size=(2,2))


# In[28]:


np.random.choice([3,5,6],p=[0.1,0.3,0.6],size=(2,2))


# In[30]:


import random
arr=np.array([1,2,3])
arr1=random.permutation(arr)


# In[31]:


arr2=random.shuffle(arr)


# In[33]:


arr


# In[35]:


arr3=np.random.permutation(arr)


# In[36]:


arr3


# In[20]:


import numpy as np


# In[21]:


arr=np.random.randint(2,9,(3,3))


# In[22]:


arr


# In[23]:


arr1=np.random.randint(7,9,(3,3))


# In[24]:


arr1


# In[25]:


arr+arr1


# In[26]:


arr+10


# In[27]:


np.add(arr,arr1)


# In[29]:


arr+(1,2,3)


# In[30]:


arr-arr1


# In[31]:


np.subtract(arr,arr1)


# In[33]:


arr1*arr


# In[34]:


np.multiply(arr,arr1)


# In[35]:


np.divide(arr,arr1)


# In[1]:


import numpy as np


# In[4]:


arr=np.random.randint(2,9,(3,3))


# In[5]:


arr


# In[6]:


arr**2


# In[7]:


arr1=np.random.randint(7,9,(3,3))


# In[8]:


arr1


# In[9]:


arr**arr1


# In[11]:


np.power(arr1,arr)


# In[ ]:





# In[22]:


np.power(arr1,2)





        
        
        


# In[2]:


import numpy as np


# In[3]:


arr1=np.random.randint(7,9,(3,3))


# In[4]:


arr1


# In[5]:


np.reciprocal(arr1)


# In[6]:


arr1


# In[7]:


arr1.T


# In[8]:


arr2=([1,2,3])


# In[10]:


np.reciprocal(arr2)


# In[12]:


np.identity(3)


# In[13]:


np.max(arr2)


# In[14]:


np.min(arr2)


# In[16]:


np.max(arr2,0)


# In[18]:


np.mean(arr2)


# In[19]:


np.median(arr2)


# In[ ]:


a=np.array([1,2,3,4,5])


# In[23]:


b=np.array([[1,2],[4,5],[6,7]])


# In[24]:


type(b)


# In[25]:


np.ndim(b)


# In[30]:


np.matrix((3,3),dtype=int)


# In[5]:


from numpy import matrix as mat


# In[6]:


a=mat([1,2,3,4,5])


# In[7]:


a


# In[8]:


import numpy as np


# In[9]:


from numpy import matrix as mat


# In[11]:


a=mat('1,2;3,4')


# In[12]:


a


# In[13]:


type(a)


# In[16]:


b=mat([[5,6],[7,8]])


# In[17]:


b


# In[18]:


type(b)


# In[20]:


mat1=mat('1,2,3,4;5,6,7,8;9,7,6,9')


# 

# In[21]:


mat1


# In[22]:


mat1.ndim


# In[23]:


mat1.size


# In[25]:


mat1.shape


# mat1.T

# In[26]:


mat1.T


# In[27]:


mat2=mat1.T


# In[28]:


mat2.base


# In[29]:


mat2+6


# In[31]:


mat1*mat2


# In[32]:


np.sqrt(mat1)


# In[33]:


np.sum(mat2)


# In[34]:


np.sum(mat2,axis=1)


# In[35]:


mat1==mat2


# In[37]:


mat1.argmax()


# In[38]:


mat2.argmin()


# In[39]:


mat2.astype(str)


# In[40]:


np.clip(mat2,3,10)


# In[45]:


np.compress([0,1,0,1],mat2,axis=0)


# In[43]:


mat2


# In[54]:


mat3=mat2.copy()


# In[55]:


mat3.fill(0)


# In[ ]:





# In[52]:


np.cumprod(mat2,axis=1)


# In[57]:


mat2.diagonal()


# In[60]:


np.flip(mat2)


# In[61]:


mat2[1:]


# In[64]:


mat2[1:, 1:]


# In[65]:


mat2.max()


# In[66]:


mat2.min()


# In[67]:


mat1.nonzero()


# In[69]:


mat2.prod()


# In[70]:


arr=mat([1,2,3])


# In[71]:


r=2


# In[72]:


mat.repeat(arr,r)


# In[ ]:




