# Introdcution
Haberman’s data set contains data from the study conducted in University of Chicago’s Billings Hospital between year 1958 to 1970 for the patients who undergone surgery of breast cancer. Source :https://www.kaggle.com/gilsousa/habermans-survival-data-set)

I would like to explain the various data analysis operation, I have done on this data set and how to conclude or predict survival status of patients who undergone from surgery.

First of all for any data analysis task or for performing operation on data we should have good domain knowledge so that we can relate the data features and also can give accurate conclusion. So, I would like to explain the features of data set and how it affects other feature.

There are 4 attribute in this data set out of which 3 are features and 1 class attribute as below. Also, there are 306 instances of data.

Number of Axillary nodes(Lymph Nodes)
Age
Operation Year
Survival Status
Lymph Node: Lymph nodes are small, bean-shaped organs that act as filters along the lymph fluid channels. As lymph fluid leaves the breast and eventually goes back into the bloodstream, the lymph nodes try to catch and trap cancer cells before they reach other parts of the body. Having cancer cells in the lymph nodes under your arm suggests an increased risk of the cancer spreading.In our data it is axillary nodes detected(0–52)
![image](https://github.com/Anup033/EDA-on-Haberman-dataset/assets/106690260/b41d8ae8-4b35-46ef-ab98-4015d0c50b6b)

Age: It represent the age of patient at which they undergone surgery (age from 30 to 83)

Operation year: Year in which patient was undergone surgery(1958–1969)

Survival Status: It represent whether patient survive more than 5 years or less after undergone through surgery.Here if patients survived 5 years or more is represented as 1 and patients who survived less than 5 years is represented as 2.

So, lets get started to play with the data set and get the conclusion.

Operations
I had used python for this purpose as it has the rich collection of machine learning libraries and mathematical operation. I will mostly use common packages as Pandas, Numpy, Matplotlib and seaborn which help me for mathematical operations and also plotting, importing and exporting of files.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
haberman= pd.read_csv(“haberman.csv”)
In above snippet using the function read_csv from pandas packages you can import the data from csv file format .So, after importing the data set you need to check if data imported properly. Below code snippet will show the shape of data that is the number of columns and rows present in data.

print(haberman.shape)

Shape of data
Here you can get confidence that the data you want get successfully imported which shows 306 instances of data and 4 attributes present as we see previously in introduction. You can even see the labels of columns present using a single line of code as follows.

print(haberman.columns)

Columns in data with type
Here the dtype means data type which is object type for all columns present in data. Similarly you can also find the dtype of feature and class attribute and can find how many number of data points belongs to 1 class and how many to other just with simple line of code as below.

haberman[“Survival_Status”].value_counts()

Number of data per classification
From above piece of code you can conclude that there are 225 patients out of 306 were survived more than 5 years and only 81 patients survived less than 5 years

Now lets plot some plots which give us more clarification of data so that we can easily come to the conclusion.

haberman.plot(kind=’scatter’, x=’Axillary_Nodes’, y=’Age’) 
plt.grid()
plt.show()
These above snippet will given me the scatter plot w.r.t the Nodes on x-axis and Age on y-axis.Our matplotlib library functions grid and show help me to plot data in grid and also to display it on console.


2D Scatter Plot
Above scatter plot shows all data in overlap fashion and also in same colour due to which we are unable to distinguish between data and also there are possibilities that you may miss some of my data which may lead to wrong conclusion. So, to distinguish between the data we can use seaborn packages function which simply to distinguish data visually by allocating different colours to every classification feature.

sns.set_style(‘whitegrid’)
sns.FacetGrid(haberman, hue=”Survival_Status”, size=4) \
 .map(plt.scatter, “Axillary_Nodes”, “Age”) \
 .add_legend();
plt.show();

2D scatter plot with different colour polarity
In above snippet, I import functions from seaborn library like FacetGrid due to which we are able to distinguish between the data classification.Here blue dots represent survival more than 5 years and orange dots represent survival less than 5 years.

As there are 3 features from which we can conclude our classification so how can we select any feature from all so that we can get output with less error rate. To do so we can use pairplots from seaborn to plot of various combination from which we can select best pair for our further operation and final conclusion. hue=”Survival_Status” will give on which feature you need to do classification. Below image shows the plotting of pairplots combination and code snippet for it.

plt.close();
sns.set_style(“whitegrid”);
sns.pairplot(haberman, hue=”Survival_Status”, size=3, vars=[‘Age’,’Operation_Age’, ‘Axillary_Nodes’])
plt.show()

Pair Plot
Above image is the combinations plot of all features in data. These types of plot is know as pairplots. Plot 1,Plot 5 and Plot 9 are the histograms of all combinations of features which explain you the density of data by considering different features of data.

Now lets take plot 1 by 1 and I will explain you that which data feature I will take for my further data analysis. I will take such a data which can show me distinguishable difference than any other data feature. So,lets start analysing each plot except plot 1,5,9 as it is a histogram of features in pairplots.

Plot 2:-In this plot you can see that there is Operation Age on X-axis and Age on Y-axis and the plot of there data is mostly overlapping on each other data so we cannot distinguish if there is any orange point present below blue point or vice versa.So I am rejecting these 2 data feature combination for further analysis.

Plot 3:-In this plot there are some points which is distinguishable but still it is better from other plot as we can provide conclusion more precisely by histogram and CDF which you will learn after a while. In this plot the overlap of points are there but still it is better than all other plots comparatively. So I will select the data feature of this plot ie. Age and Axillary nodes.

Plot 4:- It is plotted using the data feature Operation Age and Age which shows similar type of plot like Plot 2 but it just rotated by 90 degree. So I also reject this feature

Plot 6:-It plot on the feature Operation Age and Axillary nodes which is somewhat similar to the Plot 2 but overlapping of points seems to be more in this plot comparative to other. So, I will also reject this combination

Plot 7:- This plot is similar as Plot 3 only feature interchange its axis so the plot will rotate by 90 degree. Also, I will accept this combination for further operations

Plot 8:- It is same as Plot 6 only feature on axis interchange.

So, I consider the feature Age and Axillary nodes plotting in the Plot 3 and 7 for my all further data operations

1D-Scatter Plots

Lets plot 1D scatter plot and see if I can distinguish data by using below code snippet.

import numpy as np
haberman_Long_Survive = haberman.loc[haberman[“Survival_Status”] == 1];
haberman_Short_Survive = haberman.loc[haberman[“Survival_Status”] == 2];
plt.plot(haberman_Long_Survive[“Axillary_Nodes”], np.zeros_like(haberman_Long_Survive[‘Axillary_Nodes’]), ‘o’)
plt.plot(haberman_Short_Survive[“Axillary_Nodes”], np.zeros_like(haberman_Short_Survive[‘Axillary_Nodes’]), ‘o’)
plt.show()
I used the Numpy library function to plot 1D scatter plot individually for every classified data. Below you can see the 1D scatter plot using data feature Age and Axillary nodes


1D Scatter Plot
Here you can observe the data of short survival status are mostly overlap on long survival status due to which you will not able to conclude on this data.

You can get better clarification if you use PDF or CDF of data for plotting.

Let me explain you concept of PDF and CDF in high level.

PDF (Probability Density Function):- It shows the density of that data or number of data present on that point. PDF will be a peak like structure represents high peak if more number of data present or else it will be flat/ small peak if number of data present is less.It is smooth graph plot using the edges of histogram

CDF (Cumulative Distribution Function):- It is representation of cumulative data of PDF ie. it will plot a graph by considering PDF for every data point cumulatively.

Seaborn library will help you to plot PDF and CDF of any data so that you can easily visualise the density of data present on specific point.Below code snippet will plot the PDF

sns.FacetGrid(haberman,hue=”Survival_Status”, size=8)\
.map(sns.distplot,”Axillary_Nodes”)\
.add_legend()
Lets try to plot PDF of each data feature and see which data give us maximum precision.

PDF of Age


PDF for Age
Observation: In above plot it is observed that at the age range from 30–75 the status of survival and death is same. So, using this datapoint we cannot predict anything

PDF of Operation Age


PDF for Operation Age
Observation: Similar here we cannot predict anything with these histograms as there is equal number of density in each data point. Even the PDF of both classification overlap on each other.

PDF of Axillary Nodes


PDF for Axillary nodes
Observation: It has been observed that people survive long if they have less axillary nodes detected and vice versa but still it is hard to classify but this is the best data you can choose among all. So, I accept the PDF of Axillary nodes and can conclude below result

if(AxillaryNodes≤0)

Patient= Long survival

else if(AxillaryNodes≥0 && Axillary nodes≤3.5(approx))

Patient= Long survival chances are high

else if(Axillary nodes ≥3.5)

Patient = Short survival

So from above PDF we can say the patients survival status, but we cannot exactly say what percentage of patient will actually short survive or long survive. To know that we have another distribution that is CDF.

CDF will give the cumulative plot of PDF so that you can calculate what are the exact percentage of patient survival status

Let’s plot CDF for our selected feature which is Axillary nodes

counts, bin_edges = np.histogram(haberman_Long_Survive[‘Axillary_Nodes’], bins=10, 
 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
Above code will give me the CDF of Long survival status. Here we only use cumsum function from Numpy which will cumulative sum up PDF of that feature.

The CDF will of Long survival status is shown on plot in orange colour.


CDF for Long survival status
From above CDF you can observe that orange line shows there is a 85% chance of long survival if number of axillary nodes detected are < 5. Also you can see as number of axillary nodes increases survival chances also reduces means it is clearly observed that 80% — 85% of people have good chances of survival if they have less no of auxillary nodes detected and as nodes increases the survival status also decreases as a result 100% of people have less chances of survival if nodes increases >40

Let’s try to plot CDF for both feature in a single plot. To do so just add below code in existing code written for Long Survival

counts, bin_edges = np.histogram(haberman_Short_Survive['Axillary_Nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();
Below image shows the CDF for short survival in Red line


CDF for both Long and short survive
You can observe in above combine CDF for Long survival observation is same but in Short survival nearly 55% of people who have nodes less than 5 and there are nearly 100% of people in short survival if nodes are > 40

We can also predict patients status by applying mathematical formulae like Standard Deviation and Mean.

Mean is the average of all data and Standard deviation is the spread of data means how much wide the data is spread along the data set. Python have Numpy library which can perform this operation in a single line.

print(“Means:”)
print (np.mean(haberman_Long_Survive[“Axillary_Nodes”]))
print (np.mean(np.append(haberman_Long_Survive[“Axillary_Nodes”],50)))
print (np.mean(haberman_Short_Survive[“Axillary_Nodes”]))
print(“\nStandard Deviation:”)
print(np.mean(haberman_Long_Survive[“Axillary_Nodes”]))
print(np.mean(haberman_Short_Survive[“Axillary_Nodes”]))

Here we can see in line 3, I have added outlier(data which is very large or small compare to respective data. It may be an error or exception case while collecting data) even though the mean of data is not much affected.

You can observe that for Long survive mean is 2.79 and including outlier it is 3 that is almost same, but the mean of Short survive is 7.4 which is comparatively much higher than Long survive. So the probability for short survive is more in data set.

If you observe the standard deviation Long survive has standard deviation of only 2.79 and Short survive has 7.45, means the spread of data for short survive is more.

Median, Quantiles and Percentile
Some more mathematical operation you can do like Median, Quantiles, Percentile

print(“Medians:”)
print(np.median(haberman_Long_Survive[“Axillary_Nodes”]))
print(np.median(np.append(haberman_Long_Survive[“Axillary_Nodes”],50)))
print(np.median(haberman_Short_Survive[“Axillary_Nodes”]))
print(“\nQuantiles:”)
print(np.percentile(haberman_Long_Survive[“Axillary_Nodes”],np.arange(0,100,25)))
print(np.percentile(haberman_Short_Survive[“Axillary_Nodes”],np.arange(0,100,25)))
print(“\n90th percentile:”)
print(np.percentile(haberman_Long_Survive[“Axillary_Nodes”],90))
print(np.percentile(haberman_Short_Survive[“Axillary_Nodes”],90))
from statsmodels import robust
print (“\nMedian Absolute Deviation”)
print(robust.mad(haberman_Long_Survive[“Axillary_Nodes”]))
print(robust.mad(haberman_Short_Survive[“Axillary_Nodes”]))
Above code snippet will give you the Median Quantiles and nth Percentiles

Median is the centre value of data and Quantiles are the value of specific feature on nth Percentage n= 25,50,75 and nth Percentile is similar to Quantiles but n could be any number from 1 to 100.

So, for our data set we have values of these terms as follows


Observation:

From above observation it is clear that average axillary nodes in long survival is 0 and for short survival it is 4. ie, Patients who have average 4 auxillary nodes have short survival status.
Quantiles shows that nearly 50th% of axillary nodes are 0 in long survival and 75th% of patients have nodes less than 3 that is 25% patients are having nodes more than 3.
Similarly, In short survival 75th% of patients have minimum 11 nodes detected.
At 90th% there if nodes detected is >8 then it has long survival status and if nodes are >20 then patients will have short survival status
Box Plot and Whiskers, Violin plot and Contour plot
You can also analysis data using plot like Box plot Contour and more, Seaborn library has wide variety of data plotting module. Let’s take some of them

Box Plot and Whiskers

sns.boxplot(x=”Survival_Status”,y=”Axillary_Nodes”, data=haberman)
plt.show()

Box Plot and Whiskers
Here you can read this plot by observing it’s box height and width and T like structure. height of box represents all data between 25th percentile to 75th percentile and that horizontal bar represents maximum range of that data and width of box represents spread of that data in data set. Also, the small point above that vertical bar are outliers

Observation:In above box whiskers 25th percentile and 50th percentile are nearly same for Long survive and threshold for it is 0 to 7. Also, for short survival there are 50th percentile of nodes are nearly same as long survive 75th percentile. Threshold for the Short survival us 0 to 25 nodes and 75th% is 12 and 25th% is 1 or 2

So,if nodes between 0–7 have chances of error as short survival plot is also lies in it. That is 50% error for Short survival status

There are most of point above 12 lies in Short survival

Violin Plot

sns.violinplot(x=”Survival_Status”, y=”Axillary_Nodes”,data=haberman)
plt.legend
plt.show()

Violin Plot
It is same as Box whiskers plot only difference is instead of box histogram will represents spread of data.

Observation: In above violin plot we observe that For long survive density for it is more near the 0 nodes and also it has whiskers in range o-7 and in violin 2 it shows the short survival density more from 0–20 ans threshold from 0–12

Contour Plot

Contour plots are like density plot means if the number of data is more on specific point that area will get darker and if you visualise it will make hill like structure where hill top has maximum density of point and density decreases as hill slope getting decreases.

You can refer below plot how actually contour plot looks in 3D


Source: https://www.mathworks.com/help/matlab/ref/surfc.html
In above image the yellow point has maximum density.

Below is the contour plot for our data set

sns.jointplot(x=”Age”,y=”Axillary_Nodes”,data=haberman_Long_Survive,kind=”kde”)
plt.grid()
plt.show()

Contour Plot
Observation: Above is the 2D density plot for long survival using feature age and axillary nodes, it is observed the density of point for long survival is more from age range 47–60 and axillary nodes from 0–3. The dark area have major density which is hill top in 3D and density is getting low as graph get lighter. Each shade represent 1 contour plot.

Conclusion:
Yes, you can diagnose the Cancer using Haberman’s Data set by applying various data analysis techniques and using various Python libraries.

References:
https://www.mathworks.com/help/matlab/ref/surfc.html
https://www.breastcancer.org/symptoms/diagnosis/lymph_nodes
https://www.kaggle.com/gilsousa/habermans-survival-data-set
https://www.appliedaicourse.com
