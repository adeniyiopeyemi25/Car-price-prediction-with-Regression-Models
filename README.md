# Predicting car price using Regression models(Over 400k rows of data from AutoTrader)

## 📝 Introduction and Problem Statement.
Auto Trader is the largest digital automotive marketplace in the UK and Ireland, having served as the go-to destination for car buyers for the past 40 years. Its mission is to drive change together, responsibly, by expanding both car-buying and selling audiences, transforming the way cars are sold in the UK through the provision of the best online car-buying experience, and empowering all retailers to sell online. Auto Trader seeks to forge stronger partnerships with its customers, leverage its voice and influence to promote environmentally friendly vehicle choices.  

This project aims to utilize Machine Learning models to predict the price of a customer's car once it is listed on the website. The objective is to determine whether the price is overpriced, underpriced, or aligned with the current market price of the car.

## 🖇 🖥️  Prerequisites and Installation
1. Install and Upgrade scikit :-
```bash
!pip install --upgrade scikit-learn -q --user
```
2. Install SHAP :-
```bash
!pip install -q shap
```
## ⏳ Data sources
The dataset is a Car Sale Adverts dataset provided by Auto Trader, one of Manchester Metropolitan University-industry partners. Dataset was only available as a student of MMU.

## 📑 Meaning and Type of Features
- **public_reference**: The unique ID for each car advert.
- **mileage**: The total distance covered by the car to date.
- **reg_code**: This is the unique code for each year of registration.
- **standard_color**: The colour of the car.
- **standard_make**: The brand of the car.
- **standard_model**: The brand model of the car.
- **vehicle_condition**: The vehicle condition; either NEW or USED.
- **year_of_registration**: The first registration year of the car.
- **price**: The price of the car in pounds(£).
- **body_type**: The body type of the car i.e SUV, Saloon, Minibus, and so on.
- **crossover_car_and_van**: A crossover is a type of automobile with an increased ride height that is built on unibody chassis
construction shared with passenger cars, as opposed to traditional sport utility vehicles (SUV) which are built on a body-on-frame
chassis construction similar to pickup trucks.
- **fuel_type**: The type of fuel the car runs on.

## Analyzing some of the Univariate Distribution with Plots
"univariate" means that the analysis is focused on a single variable or feature. Plots are used to visualize the distribution of the variable, which can help in understanding its characteristics such as its central tendency, spread, skewness, and presence of outliers.I also have to log the price  because visualizing the logarithm of the prices using plots can help better understand the distribution and patterns in the data. 

- 💷 Analysis of Price Distribution
 
  <img src="price_distribution.png" alt="Price Distribution" width="80%" height="60%">

We can clearly see that the price has a normal distribution and the box plot at the right also shows we have some outliers as it shows from the whisker on the plot.

- 🧭 Analysis of Mileage Distribution

 <img src="mileage_distributiin.png" alt="Mileage Distribution" width="80%" height="60%">
 
 The mileage from the ileage Distribution plot shows that the mileage is righly skewed, also the boxplot shows we have some outliers.
 
-  🚌 ⛽️ Analysis of Body type and Fuel Type
  <img src="body_fuel_type.png" alt="Body/Fuel Distribution" width="90%" height="60%">
  
 Clearly it is shown here that we have more vehicles with Hatchback, followed by SUV.From the right plot we can see petrol vehicle are sold more than any other followed by the Diesel.

## ⏳ Data Processing
- #### 🚿 Dealing with Noise, Missing Values and Outliers 
    **Noise**   
The ’year of registration’ column shows some noise. The following values were observed
to be written as ’year’: 999.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1015.0, 1016.0,
1017.0, 1018.0, 1063.0, 1515.0, and 1909.0. These appear to be typos, as for instance
1006.0 should be 2006.0. I have corrected all such values. Similarly, the ’reg code’
column had some noise which was also corrected
          
   **Missing Values**  
The table below shows the columns with their corresponding number of missing values.   

| Column           | No of Missing Value |  
| -------------------|--------------------| 
| public reference   | 0                  | 
| mileage            | 127                | 
| reg code           | 31857              | 
| standard colour    | 5378               |
| standard make      | 0                  |
| standard model     | 0                  |
| vehicle condition  | 0                  |
| year of registration | 33311            |
| price              | 0                  |
| body type          | 837                |
| crossover car and van | 0               |
| fuel type          | 601                |
| log price          | 0                  |

As seen from the table, there are 6 columns with missing values. We will handle them
one at a time.
First, the NEW cars have over 30,000 with no reg code. I decided to assign 0 as it
means they are not yet registered. From the other missing reg codes, there are 608
USED cars. These were filled with their corresponding year of registration using the
[Vehicle Registration Plates of the United Kingdom](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_United_Kingdom).
Second, to fill the year of registration, I first dropped the reg codes with alphabets and
those with year of registration as NaN (less than 30 of them). I did this because I only
wanted to fill those with reg codes with numbers. Also, it is reasonable to drop those
because they are few and dropping them will not negatively affect the dataset. I filled
the year of registration using the corresponding registration plate year. For example,
if the reg code is 18, I filled the year with 2018 as seen in the Vehicle Registration
Plates of the United Kingdom.
Third, the Mileage has some missing values. I filled the mileage by grouping the
vehicles with their corresponding year of registration and filled each with the mean.
For example, all vehicles with year of registration 2012 with missing mileage were filled
with the mean of vehicles registered in the year 2012.
Fourth, the standard Color has 5344 missing values. I filled that by grouping the color
by standard model and standard make and assigning the mode of the color to the
missing values. After grouping and checking the unique values, I found out that there
are a few vehicles that were given the color 0. This happened because they are just
one particular make and model, and they do not have any color at all. I dropped those
because they are not many.
Fifth, there were 581 missing fuel types.Filling the fuel type with the mode of fuel type,
which is petrol, will not have a negative impact on our model.
The last part is filling for body type. We also grouped the body type by standard model
and standard make and assigned the mode of the body type to the missing values. Af-
ter this action, we observed that some values were given 0. This happened because
they are just one particular make and model in the dataset, and they do not have any
body type registered in the dataset. We observed that most of such vehicles are only
one except for the Volkswagen Caddy, which has 12. I checked Google to see what type
of body type it is, and I found out that it is a Wagon so I filled them with Wagon.


   **Outliers**  
It was observed that there are outliers in both the mileage and price columns. To
handle these outliers, I used the quantile method. However, since there are new cars
that should have 0 as mileage, there will not be a lower quantile. Therefore, I only
removed the upper quantile for both mileage and price.
```bash
max_threshold  = autotrader['mileage'].quantile(0.99)
autotrader = autotrader[autotrader['mileage'] < max_threshold]
autotrader[autotrader['mileage']>140000.0]
```
Below is the box plot after outliers are removed

<img src="mileage_outliers.png" alt="No Outlier Distribution" width="50%" height="50%">

- #### 🛠 Feature Engineering, Data Transformations   
Feature engineering is the process of creating new features or modifying existing ones
to improve the performance of a machine learning model.
The first feature that was engineered was vehicle age, which was achieved by subtract-
ing the year of registration from the current year. A current year column was also
created using the datetime.now() module.
Secondly, a column called average mileage was created by using a journal published on-
line titled Used Car Mileage Vs Age – Which Matters More?, According to government
statistics, the average number of miles driven by cars in England each year was 7,400
miles in 2019. Therefore, each vehicle’s average mileage was calculated by multiplying
its age with 7,400.
Lastly, the mileage was categorized into four groups. If the original mileage is between 0 and half of the average mileage, the vehicle is classified as having low mileage. If the
mileage is less than or greater than half of the average mileage, it is classified as having
good mileage. If the mileage is 15,000 or more than the average mileage, it is classified
as having high mileage. Otherwise, it is classified as having very high mileage.

## 🏽‍ Methods
- #### 📥 Feature Selection and Data Sampling
   Feature selection is the process of selecting a subset of relevant features or variables
from a larger set of features in order to improve the performance of a machine learning
model. The goal is to reduce the dimensionality of the input space while retaining as
much of the original information as possible. 
Before selectiong the features, There are some features that are naturally not relevant for the model, I will drop those once before I do automatic feature selection.
```bash
autotrader_sampled = autotrader_sampled.drop(columns=['public_reference', 'reg_code','log_price','current_year' ])
```

Also note that I reduced the sample size to 20% using the sample module so as to increase computational time.
```bash
autotrader_sampled = autotrader.sample(frac=0.20, random_state=82)
```

There are several method for feature selection like SelectKbest, Recursive Feature Elimination (RFE) and many more but I decided to use SelectKbest.Using SelectKBest requires specifying a value k to determine the best k. To accomplish this, the technique of cross-validation was employed. A linear regression model was trained with varying numbers of selected features using SelectKBest. A 10-fold cross-validation was then performed to estimate the mean squared error. The negative mean squared error was used as a scoring metric because cross val score tries to maximize the score. The mean squared error was plotted as a function of k, and the value of k
that resulted in the lowest error was selected, as shown in Figure 1. The optimal value of k was determined to be 10. Thereafter we fitted X and y with k = 10 and I am left with 10 features, It was also noticed that year of registration and vehicle age were
given similar information so one of it was dropped. The plot below shows the best k.

<img src="KBEST.png" alt="k selection plot" width="50%" height="50%">


- #### 🪚 Dimensionality Reduction
- #### 🏘 Model Building. 
  - A Linear Model
  - A Random Forest
  - A Boosted Tree
  - An Averager/Voter/Stacker Ensemble
- #### 📊 Model Evaluation and Analysis
  - Overall Performance with Cross-Validation
  - True vs Predicted Analysis
  - Global and Local Explanations with SHAP
  - Partial Dependency Plots

## 📝 Conclusion

##  📖 Acknowledgments

## Contact information

# 🏃 🚶 💃
