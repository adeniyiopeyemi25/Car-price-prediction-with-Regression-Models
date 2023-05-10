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

## Analysis of Distributions


## ⏳ Data Processing
- #### 🚿 Dealing with Missing Values, Outliers, and Noise.   
- #### 🛠 Feature Engineering, Data Transformations   

## 🏽‍ Methods
- #### 📥 Feature Selection and Data Sampling
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
