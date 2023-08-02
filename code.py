#### Problem Statement

 A Car Sale Adverts dataset provided by **AutoTrader**. I am asked to produce a regression model for predicting the selling price given the characteristics of the cars in the historical data given.

> ## <span style="color:#a1341c"> Importing Required Libraries. </span>
>
>>Importing all the necessary Library here.

!pip install --upgrade scikit-learn -q --user

!pip install -q shap
import shap
shap.initjs()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set(
    style='ticks', 
    context='talk', 
    font_scale=0.8, 
    rc={'figure.figsize': (8,5)}
)

from scipy import stats
import datetime
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,ParameterGrid
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_text

> ## <span style="color:#a1341c"> Loading/Reading and Understanding the Dataset. </span>
>
>> I am using pandas to load my data from my local hard disk using **read_csv** and I also view the data using **head** function.

autotrader = pd.read_csv('adverts.csv')

autotrader.head(1)

autotrader.columns

autotrader.shape

autotrader.info()

> ## <span style="color:#a1341c"> 1. Data/Domain Understanding and Exploration.</span>
>
>> **1.1 Meaning and Type of Features; Analysis of Univariate Distributions (3-4)**


#### Meaning and Types of Features.

From above we used  autotrader.columns which shows we have 12 columns in our dataset, 5 from the columns are explained below:  
- **Mileage** is the actual number of miles covered by the car, and it is a **Quantitative** Feature.  
- **standard_make** refers to the brand of the vehicle(Producer name), and it is a **Categorical** Feature.  
- **body_type** refers to the category of a vehicle base on its design, shape and space. and it is a **Categorical** Feature.  
- **year_of_registration** Refers to the year when the car was first registered by a user. and it is a **Quantitative** Feature.  
- **fuel_type** is the general category of fuel and can either be gasoline or natural gas. and it is a **Categorical** Feature.

#### Analysis of Univariate Distribution

The analysis of univariate distributions typically includes the examination of measures such as the mean, median, mode, variance, and skewness of the distribution, Also it will be analysed by using plots.

autotrader.drop(columns='public_reference').describe()

From above we can see the mean, median, max(mode) and standard deviation listed for mileage, year_of_registeration and price, maximum mileage in the data is 999999, mean year of registeration is 7.9 etc. We can as well determine the skewness of the above columns using Pearson's moment coefficient of skewness shown below:

mean = autotrader[['price','mileage','year_of_registration']].mean()
median = autotrader[['price','mileage','year_of_registration']].median()
std = autotrader[['price','mileage','year_of_registration']].std()
#Pearson's moment coefficient of skewness
skewness = (3 * (mean - median)) / std

skewness

Price and mileage are both positively skewed, which mean the shape of the distribution is right-skewed and year_of_registeration has a negative skew which also indicate it is left-skewed.

#### Analysing the Univariant Distribution with plots

Before ploting I decide to create a column for log_price to make my graph better.

autotrader['log_price'] = np.log(autotrader['price'])

autotrader.head(1)

#### Analysis of Price Distribution.

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Price Distribution')


sns.histplot(autotrader.log_price, ax=axs[0], color='red', label='100% Equities', kde=True, stat='density', linewidth=0)
axs[0].set_title('Price in Log')

sns.boxplot(y=autotrader.log_price, ax=axs[1])
axs[1].set_title('Car Price Spread')

plt.subplots_adjust(wspace=0.4)

#### Analysis of Mileage Distribution.

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Mileage Distribution')

sns.histplot(autotrader.dropna().mileage, ax=axs[0], color='red', label='100% Equities', kde=True, stat='density', linewidth=0)
axs[0].set_title('Mileage Distribution Plot')

sns.boxplot(y=autotrader.dropna().mileage, ax=axs[1])
axs[1].set_title('Mileage Spread')

plt.subplots_adjust(wspace=0.4)



We can clearly see that the mileage is skewed to the right, and there is some outliers showing from the boxplot.

#### Analysis of Body Type , Fuel Type and Standard colour Distribution.

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
autotrader['body_type'].value_counts(normalize=True).plot.bar()
plt.title("Count of Body Type",size=15)
plt.xlabel("Body Type")
plt.ylabel("Count")
#lt.xticks(rotation=45)

plt.subplot(1,2,2)
autotrader['fuel_type'].value_counts(normalize=True).plot.bar()
plt.title("Count of Fuel Type",size=15)
plt.xlabel("Fuel Type")
plt.ylabel("Count")
#plt.xticks(rotation=45)
plt.ylim([0, 0.6])

plt.tight_layout()



Clearly it is shown here that we have more vehicles with Hackback, followed by SUV.From the second graph Petrol has more count of vehicle followed by the Diesel

autotrader['standard_colour'].value_counts(normalize=True).plot.bar()
plt.title("Count of Colour",size=15)
plt.xlabel("Standard Colour")
plt.ylabel("Count")

Black colour vehicle is the most purchased, followed by white and grey.

> ## <span style="color:#a1341c"> 1. Data/Domain Understanding and Exploration</span>
>
>>**1.2. Analysis of Predictive Power of Features (2-3)**


autotrader.info()

#### Explore Continuous/Quantitative Features

We will have to plot the 2 countinous feature, Mileage and year_of_registeration against our target(Price).

plt.subplot(1,2,1)
sns.regplot(data=autotrader, x='mileage', y='log_price', scatter_kws={'alpha': 0.6})

plt.subplot(1,2,2)
sns.regplot(data=autotrader, x='year_of_registration', y='log_price',scatter_kws={'alpha': 0.5})

The plot above shows the importance/correlation of both mileage and year_of_registration to price, The higher the mileage the lower the price and the more recent the year of registeration is the higher the price.

#### Explore Some Categorical Features

autotrader_some_cat =['standard_colour','body_type','fuel_type']

for col in autotrader_some_cat:
    sns.catplot(x=col, y="price", data=autotrader, kind="bar", height=4, aspect=2).set(xlabel=None)
    plt.title(col)
    plt.xticks(rotation=45)

From the plot above we can see the predictive power of standard_colour,body_type,fuel_type to price.

> ## <span style="color:#a1341c"> 1. Data/Domain Understanding and Exploration</span>
>
>>**1.3. Data Processing for Data Exploration and Visualisation (1-2)**


Data processing for data exploration and visualization refers to the process of preparing, cleaning, and transforming data so that it can be effectively explored and visualized. This will be thouroughly address in the next stage (Data Processing for Maching Learning).

> ## <span style="color:#a1341c"> 2. Data Processing for Machine Learning</span>
>
>>**2.1. Dealing with Missing Values, Outliers, and Noise**


autotrader.isna().sum()

From above it shows that we have 7 from the 12 columns that has NaN(Missing Values), The columns are:
- **mileage** have **127** missing value.  
- **reg_code** have **31857** missing value.  
- **standard_colour** have **5378** missing value.  
- **year_of_registration** have **33311** missing value.  
- **body_type** have **837** missing value.  
- **fuel_type** have **601** missing value.  


**We can also see the column with the missing value from the heatmap shown below:**

sns.heatmap(autotrader.isna(),yticklabels=False)



There are different ways to determine the noise or outliers, In this case I will check the unique values of each column and check the once that does not represent the majority of each column.

#### Noise/Outliers in year_of_registration

autotrader['year_of_registration'].unique()

The **Year_of_registeration** shows some Noise, The following are seen not to be a year, it is observed that they mis typed the year, for instance 1006.0 should be 2006.0 and so on.

999.0,
 1006.0,
 1007.0,
 1008.0,
 1009.0,
 1010.0,
 1015.0,
 1016.0,
 1017.0,
 1018.0,
 1063.0,
 1515.0,
 1909.0,

#### Noise/Outliers in reg_code


autotrader['reg_code'].unique()

We will be able to know the noise on reg_code giving the link for all UK vehicle reg code with the wikipedia link:
https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_United_Kingdom

It is derived from the link that year of registeration of any vehicle is categorised into 2 registeration code depending on the month of the year the car is been registered. March to August will always have reg_code of same present year e.g vehicle registered in year 2018 will have reg_code of 18, while vehicle of same year of registeration but registered between September till February the following year will have 68 which mean 50 was added to the suppose reg_code, The alphabet also represent certain years. With this information at hand we can deduce the noise from the reg_code

The **reg_code** shows some Noise, The observed outlier are **723xuu,94,85,95,38,37 and CA**  
>- 723xuu is clearly a noise.  
>- 94, 85 and 95 are noise because they are 2044, 2035 and 2045 repectively considering the wikipedia link, we are not yet in those years.  
>- 38,37 are also noise because they should be year 2038 and 2037 respectively and we are not yet in those years.

As it has been stated above, we have 7 columns with missing value, we will have to deal with them which will be shown below:

#### Filling missing value for reg_code

autotrader[autotrader['vehicle_condition']=='NEW'].head(1)

I observed that when the vehicle_condition is NEW, we have 31,249 NaN out of the 31,857 for reg_code that are NaN. This mean the vehicle are not yet registered, I checked the data to see if there are NEW cars that has reg_code but there is none. This makes it difficut to choose a value for it. It is reasonable to fill with 0 to show that they are not registered.

autotrader.loc[autotrader['vehicle_condition']=='NEW','reg_code'] = 0

To view if NEW vehicle has been filled with the given value 0 for reg_code.

autotrader[autotrader['vehicle_condition']=='NEW'].head(1)

We can clearly see that the row with public_reference number of 202006039777689 now has reg_code of 0

autotrader['reg_code'].isna().sum()

We still have 608 rows of reg_code that are still missing which mean the vehicle_condition of those are USED, We will further find a way to fill those with corresponding year_of_registeration that are not missing.

#### Filling missing value for year_of_registeration

autotrader[autotrader['vehicle_condition']=='NEW'].head(1)

Same observation as with reg_code, when vehicle_condition is NEW, we have 31,249 NaN out of the 33,311 year_of_registeration that are NaN. After doing some rough work, I will fill the year_of registeration with current year. It mean the vehicle is NEW and has not been registered. This will also confirm NEW vehicle as reg_code 0 and year_of_registeration as any present year.

autotrader.loc[autotrader['vehicle_condition']=='NEW','year_of_registration'] = datetime.datetime.now().year

To view if the NEW vehicle have been filled with the given value of current year for our year_of_registeration, We are presently in 2023 which mean we should see 2023.

autotrader[autotrader['vehicle_condition']=='NEW'].head(1)

As we can see the first column has been filled with 2023

autotrader['year_of_registration'].isna().sum()

We still have 2062 rows of year_of_registeration that are missing, which mean the vehicle_condition of those are USED, We will further find a way to fill those with corresponding reg_code that are not missing.

#### Filling both reg_code and year_of_registeration for USED cars

Now We will have to fill reg_code and year_of_registeration with each other, which mean we will fill year_of_registeration with its corresponding reg_code that is not missing and vise versa.

There are some year_of_registeration and reg_code that are both missing and there is no way those can be filled, so its best we drop those once.

regnanYearnan=autotrader[(autotrader['reg_code'].isna())& 
                         (autotrader['year_of_registration'].isna())].index
autotrader.drop(regnanYearnan,inplace=True)

#### Filling reg_code with corresponding year_of_registeration

regCodeNaN = autotrader[(autotrader['reg_code'].isna()) & 
    (autotrader['year_of_registration'].notna())]
regCodeNaN['year_of_registration'].unique()

regCodeNaN.head(1)

The year_of_registeration of the above is 2019.0 which mean it is expected to fill the correspoinding reg_code with 19 but I am doing this for year_of_registeration that are 2001 and above, this is so because those are the once that can be filled easily considering the wikipedia UK car registeration dataset. 

for index in regCodeNaN.index:
	year_of_registration = int(regCodeNaN.loc[[index], "year_of_registration"])
	if year_of_registration >= 2001:
		#Fill with the last two digits in the year
		autotrader.loc[index, ["reg_code"]] = str(year_of_registration)[2:]

autotrader[(autotrader['reg_code'].isna()) & 
           (autotrader['year_of_registration'] >= 2001)]

autotrader['reg_code'].isna().sum()

The above are the reg_code that the year_of_Registeration are less than 2001, They are 78 and I will drop them. We can also get to fill them but I prefer to drop them since they are minimal.

regnan=autotrader[(autotrader['reg_code'].isna())& 
                  (autotrader['year_of_registration'].notna())].index

autotrader.drop(regnan,inplace=True)

autotrader['reg_code'].isna().sum()

We have finally filled all the reg_code.

#### Filling year_of_registeration with corresponding reg_code

To fill the year of registeration I will have to drop the following:

>- Reg_code with noise (Just 9 of them).
>- Reg_code with alphabet that has year_of_registeration as NAN(less than 30 of them).

I am doing this because I only want to fill those that has reg_code with numbers, Also it is reasonable to drop those because they are few and droping them will not affect my dataset negatively. It can also be filled but this will take time because I will have to code for each alphabet.
 

autotrader[autotrader['reg_code'].isin(['94', '85', 'CA',  '723xuu', '95', '38',  '37'])].head(1)

noise=autotrader[autotrader['reg_code'].isin(['94', '85', 'CA',  '723xuu', '95', '38',  '37'])].index
autotrader.drop(noise , inplace=True)

#Observing the reg_code with alphabet and year_of_registeration that us not missint.
autotrader[
         (autotrader['reg_code'].isin(['B', 'P', 'E','R', 'L', 'C', 'Y','M', 'J', 'H','S','N', 'F', 'T', 'V','G', 'D', 'A']))&
         (autotrader['year_of_registration'].isna())
].head(1)

#Giving the above code a variable name Regcodeaphabet and I will drop those rows.
Regcodeaphabet=autotrader[
         (autotrader['reg_code'].isin(['B', 'P', 'E','R', 'L', 'C', 'Y','M', 'J', 'H','S','N', 'F', 'T', 'V','G', 'D', 'A']))&
         (autotrader['year_of_registration'].isna())
].index
autotrader.drop(Regcodeaphabet, inplace=True)

#I finally have a clean reg_code and year_of_registeration that are clean.
yearORegNaNnew = autotrader[
    (autotrader['year_of_registration'].isna()) & 
    (~autotrader['reg_code'].isna())
]
yearORegNaNnew['reg_code'].unique()

for index in yearORegNaNnew.index:
    reg_code = yearORegNaNnew.loc[[index], "reg_code"].tolist()[0]
    if int(yearORegNaNnew.loc[[index], "reg_code"]) > 50:
        # Subtract 50 from reg_code if it's greater than 50 and add 2000
        autotrader.loc[[index], "year_of_registration"] = 2000 + (int(reg_code) - 50)
    else:
        # Add 2000 to reg_code if it's less than 50
        autotrader.loc[[index], "year_of_registration"] = 2000 + int(reg_code)

autotrader['year_of_registration'].isna().sum()

We have finally filled all the year_of_registeration.

**Now I want to check if we still have the noise for both reg_code and year_of_registeration, If available we will replace them with the right one.**

autotrader['reg_code'].unique()

'94', '85', 'CA',  '723xuu', '95', '38',  '37', This were our noise in reg_code, we dropped them above and they are no longer showing.

autotrader['year_of_registration'].unique()

999.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1015.0, 1016.0, 1017.0, 1018.0, 1063.0, 1515.0, 1909.0, This were the noise we found in the year_of_registeration, as we can see we still have them in our data so we will have to replace them. We will replace them with 1999.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2015.0, 2016.0, 2017.0, 2018.0, 2003.0, 2015.0, 2009.0


#Replacing the noise above with its corresponding correct once
autotrader['year_of_registration'].replace(
    {999.0:1999.0, 1006.0:2006.0, 1007.0:2007.0, 1008.0:2008.0, 1009.0:2009.0, 1010.0:2010.0, 1015.0:2015.0, 1016.0:2016.0, 1017.0:2017.0, 1018.0:2018.0, 1063.0:2003.0, 1515.0:2015.0, 1909.0:2009.0},inplace=True
)

autotrader['year_of_registration'].unique()

The noise in the year_of_registeration are no more.

autotrader.isna().sum()

autotrader.shape

At this point we have only drop limited amount of rows, We have 402005 from the start and now we have 401549 rows.

#### Filling missing value in mileage

We will fill the mileage by grouping the vehicle with their corresponding year_of_registeration and fill with each mean. For example, all vehicle with year_of_registeration 2012 with missing mileage will be filled with the mean of vehicle registered in year 2012 

autotrader['mileage']=autotrader.groupby('year_of_registration', group_keys=False)['mileage'].apply(lambda x:x.fillna(x.mean()))

autotrader['mileage'].isna().sum()

We have our mileage also finally filled

#### Filling missing value for standard_colour

autotrader['standard_colour'].isna().sum()

From the above we have 5344 Nan Standard_colour, There are 2 columns that are correlated to it,also with my domain knowledge I think its right to group colour by standard_model and standard_make and fill the mode of colour.

autotrader['standard_colour'] = autotrader.groupby(['standard_make', 'standard_model'], group_keys=False)['standard_colour'].apply(lambda x: x.fillna(stats.mode(x)[0][0]))


autotrader['standard_colour'].unique()

After groupby and checking the unique value, I found out that there are few of the vehicle that were given color 0, This happened because they are just 1 particular make and model and they do not have any color at all, this will be shown below:

autotrader[autotrader['standard_colour']==0].head(1)

Its either I decide a colour for this once or I drop them, I decided to drop them because they are not much.

autotrader=autotrader[autotrader['standard_colour']!=0]

autotrader['standard_colour'].isna().sum()

We have our standard_colour finally filled

#### Filling missing value for fuel_type

autotrader['fuel_type'].isna().sum()

We have 581 fuel_type that are missing out of over 400k data, filling the fuel type with mode of fuel_type which is petrol will not have a negative impact on our model.

fuel_type_mode = autotrader['fuel_type'].mode()[0]

autotrader['fuel_type'] =autotrader['fuel_type'].fillna(fuel_type_mode)

autotrader['fuel_type'].isna().sum()

We have also filled the fuel_type.

#### Filling missing value for Body_Type.

autotrader['body_type']=autotrader.groupby(['standard_make','standard_model'], group_keys=False)['body_type'].apply(lambda x:x.fillna(stats.mode(x)[0][0]))

autotrader['body_type'].unique()

After groupby and checking the unique value, I found out that there are few of the vehicle that were given body_type 0, This happened because they do not have a body type for the group of make and model this will be shown below:

no_body_type = autotrader[autotrader['body_type']==0]
no_body_type[['standard_make','standard_model','body_type']].value_counts()

From the above we have a vehicle that is Volkswagen Caddy that are 12 in numbers, I will fill it. I check google to see what type of body type is it and I found out it is Wagon, I can as well do that for others but because they are 1 or 2 in number, I will drop them.

autotrader.loc[(autotrader['standard_make'] == 'Volkswagen') 
               & (autotrader['standard_model'] == 'Caddy'), 'body_type'] = 'wagon'

autotrader[autotrader['standard_model'] == 'Caddy'].head(1)

We can confirm that this has also been filled with body_type Wagon. I will now drop the other once.

autotrader=autotrader[autotrader['body_type']!=0]

autotrader[autotrader['body_type'] == 0]

autotrader['body_type'].isna().sum()

We have finally filled and clean the body_type.

#### Removing Noise/Outlier from Mileage

max_threshold  = autotrader['mileage'].quantile(0.999)
max_threshold 

I am setting my quantile to 0.9999, due to my domain knowledge about cars, any car above 196050.48600000003 mileage is said to be an outlier and from the data we can confirm as those figures above this does not represent the majority of the mileage data.

autotrader = autotrader[autotrader['mileage'] < max_threshold]

autotrader[autotrader['mileage']>196050.48600000003]

We have taken any mileage above 196050.48600000003 from our dataset.

sns.boxplot(x="mileage",data=autotrader)

The box plot above shows we have reduced/removed our outliers from the mileage, this is so because there are no point outside the whisker.

#### Removing Noise/Outlier from Price

max_threshold  = autotrader['price'].quantile(0.976)

max_threshold 

I am setting my quantile to 0.98, due to my domain knowledge about cars, any car above 66750.0 pounds  is said to be an outlier and from the data we can confirm as those figures above this does not represent the majority of the mileage data. Also it is been confirm by AUTOTRADER that they do not set price for cars above this value.

autotrader = autotrader[autotrader['price'] < max_threshold]

autotrader[autotrader['price']>66750.0]

sns.boxplot(x="price",data=autotrader)

The box plot above shows we have reduced/removed our outliers from the price, this is so because there are few points outside the whisker.

autotrader.shape

autotrader.isna().sum()

We are done filling the missing value and We still have 400711 rows from the original 402005, which mean our sample is still preserved.

#### Visualisation after Data Exploration.

**Plot mileage vs log_price.**

sns.regplot(data=autotrader, x='mileage', y='log_price', scatter_kws={'alpha': 0.6})

It is clear that the higher the mileage the lower the price.

**body_type vs log_price.**

plt.title('Body Type vs Price')
plt.xticks(rotation=45)
sns.boxplot(x=autotrader.body_type, y=autotrader.log_price, palette=("plasma"))
plt.ylabel("Price in Log")

The Body Type SUV have higher price range than the others(though some has some high values outside the whiskers.). Most of the body type has normal distribution except for Limousine and MPV. Limousine,Window Van, Camper, Car Derived Van and Chassis Cab do not have any outlier.

**body_type vs fuel_type.**

plt.title('Body Type vs Fuel Type')
plt.xticks(rotation=90)
sns.histplot(data=autotrader, x='body_type', kde=True, hue='fuel_type')

We can see that:  
1) SUV has both higher number of diesel and electric than any other type of fuel type.  
2) Hatchback has the highest number of petrol than any other body type and also have some that are electric.  

3) Most of the body type of car seem to have electric fuel type.

> ## <span style="color:#a1341c">2. Data Processing for Machine Learning</span>
>
>> **Feature Engineering, Data Transformations, Feature Selection (2-3)**


#### Feature Engineering

First I will like to know the age of all vehicles, This will be year_of_registeration subtracted from the current_year.

#Creating a new column current_year
autotrader['current_year'] = datetime.datetime.now().year

autotrader.head(1)

#Creating a new column called vehicle age.
autotrader['vehicle_age']= autotrader['current_year'] - autotrader['year_of_registration']

autotrader.head(1)

#### Worthy to NOTE:

Referecing to an jornal online(Used Car Mileage Vs Age – Which Matters More? https://news.motors.co.uk/used-car-mileage-vs-age-which-matters-more)

Calculating the average mileage of a car is tricky. According to government statistics, the average number of miles driven by cars in England each year was 7,400 miles in 2019 (the latest figures are for 2020, but they’re skewed downwards due to pandemic lockdowns so we’ve based the average on 2019 figures as it was a more normal year).

With the above claim I will like to get the average mileage of the vehicle.

autotrader['average_mileage'] = autotrader['vehicle_age'] * 7400

autotrader.head(1)

With the average_mileage it will not be a bad idea to categorise our vehicle considering the original mileage versus the average mileage. Say if the original mileage is between 0 to half of the average mileage, the vehicle has Low mileage,if mileage is less than and also greater that half of average mileage categorize as Good mileage, if mileage is 15000 greater than average mileage, categorize as High mileage, else Very High mileage

def categorize_mileage(autotrader):
    mileage = autotrader['mileage']
    average_mileage = autotrader['average_mileage']
    if mileage == 0 or mileage < average_mileage / 2:
        return "Low mileage"
    elif mileage < average_mileage:
        return "Good mileage"
    elif mileage < average_mileage + 15000:
        return "High mileage"
    else:
        return "Very High mileage"

autotrader['mileage_category'] = autotrader.apply(categorize_mileage, axis=1)

autotrader.head(1)

autotrader.shape

The column was originaly 12 but I have added five more features.

autotrader_main = autotrader.copy()

autotrader_main.head(3)

autotrader.info()

### DATA TRANSFROMATION

#### 20% Sample taken

#Taking 20% sample of all data
autotrader_sampled = autotrader.sample(frac=0.20, random_state=82)

autotrader_model = autotrader_sampled.copy()

autotrader_explain = autotrader_model.copy()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from functools import partial
rmse = partial(mean_squared_error, squared=False)


import warnings
warnings.filterwarnings("ignore")


### AUTOMATED FEATURE SELECTION USING SelectKBest

autotrader_sampled.head()

autotrader_sampled.info()

autotrader_sampled = autotrader_sampled.drop(columns=['public_reference', 'reg_code','log_price','current_year' ])

X, y = autotrader_sampled.drop(columns='price'), autotrader_sampled['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

numeric_features = X.select_dtypes(exclude='object').columns.tolist()
numeric_transformer = Pipeline(
    steps=[
        #("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ]
).set_output(transform='pandas')

from category_encoders import TargetEncoder

categorical_features = X.select_dtypes(include='object').columns.tolist()
categorical_transformer = Pipeline(
    steps=[        #("imputer", SimpleImputer(strategy="most_frequent")),         
                    ("te", TargetEncoder()),    ]
).set_output(transform='pandas')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")

preprocessor.fit(X, y)

X_transformed = preprocessor.transform(X)

X_transformed.head(3)

Selecting k best.  
Plotting negative mean squared error versus the range of k values.    
The lowest point on the curve indicates the best k value.

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression

# Define the range of k values to try
k_values = range(1, X_transformed.shape[1]+1)

# Perform cross-validation for each k value
cv_scores = []
for k in k_values:
    # Create a SelectKBest transformer with k features
    selector = SelectKBest(f_regression, k=k)
    # Transform the data using the selector
    X_selected = selector.fit_transform(X_transformed, y)
    # Train a linear regression model using cross-validation
    reg = LinearRegression()
    scores = cross_val_score(reg, X_selected, y, cv=10, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Plot the mean squared error as a function of k
import matplotlib.pyplot as plt
plt.plot(k_values, cv_scores)
plt.xlabel('k')
plt.ylabel('Negative MSE')
plt.show()


selector = make_pipeline(
    SelectKBest(f_regression, k=10).fit(X_transformed, y)
).set_output(transform='pandas')

selector.get_feature_names_out()

X_sel = selector.transform(X_transformed)

X_sel.head()

Dropping year_of_registration because it has similar information with "vehicle_age"

X_sel = X_sel.drop(columns = 'year_of_registration')

X_sel.head()

X_sel.columns



### ENCODING THE NEEDED COLUMN

autotrader_model.head()

#### Drop the column not represented from our Selectkbest Feature selection.

autotrader_model = autotrader_model.drop(columns=['public_reference', 'reg_code','standard_colour','crossover_car_and_van','log_price','current_year','year_of_registration' ])

autotrader_model.info()

all_cat_columns = ['standard_make','standard_model', 'vehicle_condition', 'body_type', 'fuel_type','mileage_category']

autotrader_model[all_cat_columns].nunique()

**One hot encoding** for columns with categories not more than 4, vehicle_condition and mileage_category

some_cat_feat = [ 'vehicle_condition','mileage_category']
autotrader_sampled_encoded= pd.get_dummies(autotrader_model, columns=some_cat_feat)

**Target encoding** is used for columns with category more than 4,standard_make, standard_model, body_type and fuel_type.

import category_encoders as ce

# define the columns to encode
columns_to_encode = ['standard_make', 'standard_model', 'body_type','fuel_type']

# create a target encoder instance
target_encoder = ce.TargetEncoder(cols=columns_to_encode)

# fit the target encoder to your data
target_encoder.fit(autotrader_model[columns_to_encode], autotrader_sampled_encoded['price'])

# transform the encoded columns and replace them in your original DataFrame
autotrader_sampled_encoded[columns_to_encode] = target_encoder.transform(autotrader_sampled_encoded[columns_to_encode])


autotrader_sampled_encoded.head()

autotrader_nopca = autotrader_sampled_encoded.copy()

### DIMENTIONALITY REDUCTION (PCA)

#Drop Price for PCA 
autotrader_sampled_encoded = autotrader_sampled_encoded.drop(columns = 'price')

autotrader_sampled_encoded.columns



elbow method" where you plot the explained variance ratio as a function of the number of components and look for the point where the curve starts to level off or plateau

#Scale before PCA
scaler = StandardScaler()
scaler.fit(autotrader_sampled_encoded)
X_scaled = scaler.transform(autotrader_sampled_encoded)

pca_full = PCA()
pca_full.fit(X_scaled)

pca_full.components_

pca_full.explained_variance_ratio_

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# keep the first ten principal components of the data
pca = PCA(n_components=8)
# fit PCA model to autotrader dataset
pca.fit(X_scaled)

# transform data onto the first ten principal components
autotrader_sampled_encoded = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(autotrader_sampled_encoded.shape)))

# Create a matshow plot of the first 7 principal components
plt.matshow(pca.components_[:8,:], cmap='coolwarm')
plt.yticks(np.arange(8), ["Component"+" "+str(i+1) for i in range(8)])
plt.colorbar()

# Set the x-axis ticks to be the feature names
feature_names = ['mileage', 'standard_make', 'standard_model', 'body_type',
       'fuel_type', 'vehicle_age', 'average_mileage', 'vehicle_condition_NEW',
       'vehicle_condition_USED', 'mileage_category_Good mileage',
       'mileage_category_High mileage', 'mileage_category_Low mileage',
       'mileage_category_Very High mileage']
plt.xticks(range(len(feature_names)), feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")


autotrader_pca = pd.DataFrame(autotrader_sampled_encoded, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5','Component 6','Component 7','Component 8'])


autotrader_pca.head()

### MODEL BUILDING

#split into training/test sets
X = autotrader_pca
y = autotrader_sampled["price"]
X.shape,y.shape

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=1000)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

models = pd.DataFrame(columns=["Model","MAE","MSE","RMSE","R2 Score"])
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse
    

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared


### POLYNOMIAL REGRESSION WITH GRID SEARCH

Scaling done on PCA already so not needed for polynomial again

poly2 = Pipeline(steps=[
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('regr', Ridge())
]).set_output(transform='pandas')

param_grid = dict(
    regr__alpha=[0.001, 0.01, 0.1, 1, 10, 100],
    polynomial__degree=[2, 3, 4]
)

poly2_grid = GridSearchCV(
    poly2,
    param_grid,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    cv=5
)

poly2_grid.fit(X_train, y_train)

poly2_grid_results = pd.DataFrame(poly2_grid.cv_results_)
poly2_grid_results.columns

poly2_grid_results[[
    'param_polynomial__degree', 'param_regr__alpha', 
    'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score',
    'rank_test_score'
]].sort_values('rank_test_score')

poly2_model = poly2_grid.best_estimator_

# make predictions
predictions = poly2_model.predict(X_test)

# evaluate the model
mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)


# add the results to the models dataframe
new_row = {"Model": "PolynomialRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
models = pd.concat([models, pd.DataFrame(new_row, index=[0])], ignore_index=True)




### RANDOM FOREST REGRESSION WITH GRID SEARCH

rf = RandomForestRegressor(random_state=1000)

param_grid_rfr = {
    'n_estimators': [25, 50, 75, 100,125,150],
    'max_depth': [2, 3, 4, 5],
}

grid_rfr = GridSearchCV(
    rf,
    param_grid_rfr,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    cv=5
)

grid_rfr.fit(X_train, y_train)

grid_rfr_results = pd.DataFrame(grid_rfr.cv_results_)
grid_rfr_results.columns

grid_rfr_results[
    ['param_max_depth','param_n_estimators', 'mean_train_score', 'std_train_score',
     'mean_test_score','std_test_score', 'rank_test_score'  ] 
].sort_values('mean_test_score', ascending=False)

rf_model = grid_rfr.best_estimator_

# make predictions
predictions = rf_model.predict(X_test)

# evaluate the model
mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)

# add the results to the models dataframe
new_row = {"Model": "RandomForestRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
models = pd.concat([models, pd.DataFrame(new_row, index=[0])], ignore_index=True)



### GRADIENT BOOSTING REGRESSOR WITH GRID SEARCH

gb = GradientBoostingRegressor(random_state=1000)

param_grid_gb = {
    'n_estimators': [25, 50, 75, 100,125],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 3, 4]
}

grid_gb = GridSearchCV(
    gb,
    param_grid_gb,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    cv=5,
    n_jobs=-1, 
    verbose=2
)

grid_gb.fit(X_train, y_train)

grid_gb_results = pd.DataFrame(grid_gb.cv_results_)
grid_gb_results.columns

grid_gb_results[
    ['param_n_estimators','param_learning_rate','param_max_depth', 'mean_train_score', 'std_train_score',
     'mean_test_score', 'std_test_score', 'rank_test_score'  ] 
].sort_values('mean_test_score', ascending=False)

gb_model = grid_gb.best_estimator_

predictions = gb_model.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)

new_row = {"Model": "GradientBoostingRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
models = pd.concat([models, pd.DataFrame(new_row, index=[0])], ignore_index=True)




### ESEMBLE(VotingRegressor)

ensembled_model = [poly2_model, rf_model, gb_model ]

for est in ensembled_model:
    est.fit(X_train, y_train)

ensemble = VotingRegressor(
    [
        ("poly2", poly2_model), 
        ("rf", rf_model), 
        ("gb", gb_model)
    ]
)
ensemble.fit(X_train, y_train)

all_regr = ensembled_model + [ ensemble ]

predictions = ensemble.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)

new_row = {"Model": "Esemble(VotingRegressor)","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
models = pd.concat([models, pd.DataFrame(new_row, index=[0])], ignore_index=True)


### Overall Performance with Cross-Validation USING RMSE and MAE

for est in all_regr:
    scores = cross_val_score(est, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    print(scores.mean()*-1, scores.std())

The overall performance with cross-validation is the average root mean squared error (RMSE) across all the folds of the cross-validation. From the results you above, the overall performance of the four models is:  

Polynomial Regression: 4367  
RandomForest Regression: 5058   
Gradient Boosting Regression: 4222  
Essemble(VotingRegressor): 4323 

Lower RMSE values indicate better performance, so it looks like the Gradient Boosting Regression model has the best overall performance among the four models tested.  

The standard deviation of the RMSE scores provides information about the variability of the model's performance across different folds of the data. A lower value of the standard deviation suggests that the model is less sensitive to the choice of training and validation sets.

for est in all_regr:
    scores = cross_val_score(est, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(scores.mean()*-1, scores.std())

### True vs Predicted Analysis Using Gradient boost which is the best model

y_test

y_pred = gb_model.predict(X_test)

# convert y_pred and y_test to pandas Series
y_pred_series = pd.Series(y_pred, name='predicted_price')
y_test_series = pd.Series(y_test, name='true_price')

# reset the indices to be consecutive integers
y_pred_series.reset_index(drop=True, inplace=True)
y_test_series.reset_index(drop=True, inplace=True)

# concatenate the two Series into a DataFrame
result = pd.concat([y_test_series, y_pred_series], axis=1)

# compute the difference between predicted and true prices
result['price_diff'] = np.abs(result['predicted_price'] - result['true_price'])


result.sample(5)

np.mean(result['price_diff'])

# plot regplot using seaborn
sns.regplot(x='true_price', y='predicted_price', data=result)

# set title and labels for the plot
plt.title('True Price vs Predicted Price')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')

Ideally, all points should be close to a regressed diagonal line. So, if the Actual is 5, predicted should be reasonably close to 5.  
For perfect prediction, Predicted=Actual, or 𝑥=𝑦, so the graph shows how much the prediction deviated from actual value (the prediction error).

In the graph, the prediction was mostly overestimating the actual outcome (𝑦>𝑥)

The points that are tightly clustered around the diagonal line indicates that the model has high accuracy and is making good predictions. On the other hand, the points that are scattered widely, indicate that the model is not performing well.

## SHAP, ICE AND PDP

autotrader_model.head()

autotrader_model.info()

X_model = autotrader_model.drop(columns = 'price')
y_model = autotrader_model['price']

X_model.columns

X_model_train, X_model_test, y_model_train, y_model_test = train_test_split(X_model, y_model, test_size=1/4, random_state=0)

numeric_features = X_model.select_dtypes(exclude='object').columns.tolist()
numeric_transformer = Pipeline(
    steps=[
        #("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ]
).set_output(transform='pandas')

from category_encoders import TargetEncoder

categorical_features = X_model.select_dtypes(include='object').columns.tolist()
categorical_transformer = Pipeline(
    steps=[        #("imputer", SimpleImputer(strategy="most_frequent")),         
                    ("te", TargetEncoder()),    ]
).set_output(transform='pandas')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")

from sklearn.ensemble import HistGradientBoostingRegressor
regr_model = HistGradientBoostingRegressor(
    max_depth=8, max_iter=400
)


regr_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regr", regr_model)
    ]
).set_output(transform="pandas")

regr_pipe.fit(X_model_train, y_model_train)

from sklearn.metrics import mean_squared_error
from functools import partial
rmse = partial(mean_squared_error, squared=False)
rmse(y_model_test, regr_pipe.predict(X_model_test))

## Predicting with histgradientboosting

y_pred = regr_pipe.predict(X_model_test)

y_model

def plot_true_vs_predicted(
        est, 
        X_train, y_train,
        X_test, y_test,
        ax=None,
        train_style_kws={},
        test_style_kws={}
    ):
    if ax is None:
        fig, ax = plt.subplots()
    y_pred_train = regr_pipe.predict(X_model_train)
    y_pred_test = regr_pipe.predict(X_model_test)
    ax.plot(y_train, y_pred_train, '.', label='train', **train_style_kws)
    ax.plot(y_test, y_pred_test, '.', label='test', **test_style_kws)
    ax.set_xlabel('True Target')
    ax.set_ylabel('Predicted Target')
    # the diagnonal line for the idealised space of predictions
    ax.plot(
        [0, 1], [0, 1], transform=ax.transAxes, 
        color='gray', linestyle=':', alpha=0.3
    )
    ax.legend()

    return ax

fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
plot_true_vs_predicted(
    regr_pipe,
    X_train, y_train,
    X_test, y_test, 
    ax=ax
);

y_pred_train = regr_pipe.predict(X_model_train)
y_pred_test = regr_pipe.predict(X_model_test)
# convert y_pred and y_test to pandas Series
y_pred_series = pd.Series(y_pred_train, name='predicted_price')
y_test_series = pd.Series(y_pred_test, name='true_price')

# reset the indices to be consecutive integers
#y_pred_series.reset_index(drop=True, inplace=True)
#y_test_series.reset_index(drop=True, inplace=True)

# concatenate the two Series into a DataFrame
#result = pd.concat([y_test_series, y_pred_series], axis=1)

# compute the difference between predicted and true prices
result['price_diff'] = np.abs(result['predicted_price'] - result['true_price'])


result.sample(10)

### Saving file name

import pickle

filename = "train_model.sav"

pickle.dump(regr_pipe, open(filename, 'wb'))













X_model_encoded = preprocessor.fit_transform(X_model_train, y_model_train)
print(X_model_encoded.head())


### SHapley Additive exPlanations (SHAP)

explainer = shap.TreeExplainer(regr_pipe['regr'])

explainer = shap.TreeExplainer(regr_model)

shap_values = explainer.shap_values(
    regr_pipe['preprocessor'].transform(X_model)
)

shap.summary_plot(
    shap_values, 
    regr_pipe['preprocessor'].transform(X_model),
    max_display=8
)



The sharp summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value of an instance per feature. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value of each instance. The combined feature standard model and standard make are the most important feature, has a high Shapley value range. The color represents the value of the feature from low to high. Overlapping points are jittered in the y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance.  

Feature importance: Features(variable) are ranked in descending order.  
Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.  
Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.  
Correlation: A high level of the “mileage and standard make” has a high and positive impact on the price and it is negatively correlated. The “high” comes from the red color, and the “positive” impact is shown on the X-axis. 

#### Seeing one feature with pdp

from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    regr_pipe, X_model, features=['mileage'], kind='both'
);



#### I know I can view PDP without ICE but will like to see what ICE for mileage is

### Individual Conditional Expectation (ICE)

X_model['mileage'].min(), X_model['mileage'].max()

mil_synth_data = np.linspace(X_model['mileage'].min(), X_model['mileage'].max(), 100)
mil_synth_data[:5], mil_synth_data[-5:]

an_instance = X_model.sample(1, random_state=42).drop(columns='mileage')
an_instance

temp_synth_df = pd.DataFrame(mil_synth_data, columns=['mileage'])
temp_synth_df.head()

synth_df = an_instance.merge(temp_synth_df, how='cross')
print(synth_df.shape)
synth_df

pred = regr_pipe.predict(synth_df)
pred[:5], pred[-5:]

sns.displot(X_model['mileage'], kde=True, rug=True);

ax = sns.lineplot(x=synth_df['mileage'], y=pred);
ax.set_ylim(0, 50000);

### Partial Dependency Plot (PDP)

PartialDependenceDisplay.from_estimator(
    regr_pipe, X_model_test, features=['mileage'], kind='both'
);

fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
PartialDependenceDisplay.from_estimator(
    regr_pipe, X_model_test, features=X_model_test.select_dtypes(exclude='object').columns,
    kind='both',
    subsample=500, grid_resolution=30, n_jobs=2, random_state=0,
    ax=ax, n_cols=2
);



**Checking just the average of the non object datatypes**

fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
PartialDependenceDisplay.from_estimator(
    regr_pipe, X_model_test, features=X_model_test.select_dtypes(exclude='object').columns,
    kind='average',
    subsample=500, grid_resolution=30, n_jobs=2, random_state=0,
    ax=ax, n_cols=2
);

**The average of 2 features and their interaction**

fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
PartialDependenceDisplay.from_estimator(
    regr_pipe, X_model_test, 
    features=["mileage", "vehicle_age", ("mileage", "vehicle_age")],
    kind='average', 
    ax=ax, n_cols=3,
    subsample=500, grid_resolution=30, n_jobs=2, random_state=0,
);





# EXTRA.... HOW ABOUT WITHOUT PCA

I am doing this for my own understanding, just curious why reducing the dimention?

autotrader_nopca.head()

X_nopca = autotrader_nopca.drop(columns = 'price')
y_nopca = autotrader_nopca['price']

X_nopca.train,X_nopca.test, y_nopca.train,y_nopca.test = train_test_split(X_nopca, y_nopca, test_size=1/4, random_state=0)

poly2_npoca = Pipeline(steps=[
    ('scaling', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('regr', Ridge())
]).set_output(transform='pandas')

param_nopca = dict(
    regr__alpha=[0.001, 0.01, 0.1, 1, 10, 100],
    polynomial__degree=[2, 3, 4]
)

poly2_grid = GridSearchCV(
    poly2_npoca,
    param_nopca,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    cv=5
)

poly2_grid.fit(X_nopca.train, y_nopca.train)

poly2_grid_results = pd.DataFrame(poly2_grid.cv_results_)
poly2_grid_results.columns

grid_rfr_results[
    ['param_max_depth','param_n_estimators', 'mean_train_score', 'std_train_score',
     'mean_test_score','std_test_score', 'rank_test_score'  ] 
].sort_values('mean_test_score', ascending=False)

poly2_model = poly2_grid.best_estimator_

# make predictions
predictions = poly2_model.predict(X_nopca.test)

# evaluate the model
mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)


# add the results to the models dataframe
new_row = {"Model": "PolynomialRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared}
models = pd.concat([models, pd.DataFrame(new_row, index=[0])], ignore_index=True)


### WOW, I CAN CONFIRM THE MODEL WITH PCA HAVE A BETTER SCORE


