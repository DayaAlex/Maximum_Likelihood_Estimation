import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sample = norm.rvs(loc=0 ,scale=1, size=50) #Generating random samples from a normal distribution with mean 0, standard deviation =1
print('generated sample data:',sample)

MLE_mean, MLE_standard_deviation = norm.fit(sample)#find actual mean and variance generated data
print("MLE mean and MLE variance:", MLE_mean,"&",MLE_standard_deviation**2)

#plotting MLE for mean
range_of_x=np.linspace(-2,2,500)#chooses range for mean
mean_likelihood =np.ones(500)#generate an array of 1
for i in sample:
	mean_likelihood *=norm(range_of_x,1).pdf(i)#fix std=1, find each likelihood corresponding to range of mean in the sample pdf, i.e P(data|mean)

#mean_MAP= mean_likelihood*norm(new,beta_square)


fig, ax= plt.subplots(figsize=(12,6))
plt.xlabel('MLE mean')
plt.ylabel('P(data|MLE mean)')
ax.plot(range_of_x,mean_likelihood)#plotting the likelihood given the MLE parameter of mean
plt.axvline(MLE_mean,color='r', label= 'MLE mean')
plt.axvline(0,color='b', label='TRUe MEAN =0')
plt.legend()
plt.show()# we can see that likelihood is maximum for MLE mean and not True mean

#plotting MLE for variance
range_of_x=np.linspace(0,2,500)#chooses range for std
std_likelihood =np.ones(500)
for i in sample:
	std_likelihood *=norm(0,range_of_x).pdf(i)#fix mean=0, find likelihood corresponding to range of std in the sample pdf, i.e P(data|std)

fig, ax= plt.subplots(figsize=(12,6))
plt.xlabel("MLE variance")
plt.ylabel("P(data|MLE variance)")
ax.plot(range_of_x**2,std_likelihood)
plt.axvline(MLE_standard_deviation**2,color='r', label='MLE variance')
plt.axvline(x=1,color='b',label='TRUE variance')
plt.legend()
plt.show()#we see that likelihood is maximum for MLE variance and not True variance

#plotting actual distribution and estimated distribution
#fig, ax=plt.subplots(figsize=(12,6))
#plt.