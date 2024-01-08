Health Insurance is a type of insurance that covers medical expenses. A person who has taken a health insurance policy gets health insurance cover by paying a particular premium amount. <p>


# Health Insurance Premium Prediction <p>
The amount of the premium for a health insurance policy depends from person to person, as many factors affect the amount of the premium for a health insurance policy. Let’s say age, a young person is very less likely to have major health problems compared to an older person. Thus, treating an older person will be expensive compared to a young one. That is why an older person is required to pay a high premium compared to a young person.Just like age, many other factors affect the premium for a health insurance policy. <p>

# Health Insurance Premium Prediction using Python <p>
The dataset that I am using for the task of health insurance premium prediction is collected from Kaggle. It contains data about: <p>

1. the age of the person  <p>
2. gender of the person <p>
3. Body Mass Index of the person <p>
4. how many children the person is having <p>
5. whether the person smokes or not <p>
6. the region where the person lives <p>
7. the charges of the insurance premium <p>

**So let’s import the dataset and the necessary Python libraries that we need for this task:** <p>

import numpy as np <p>
import pandas as pd <p>
data = pd.read_csv("Health_insurance.csv") <p>
data.head() <p>
![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/f49231e4-da39-4b85-ace9-36341c7821b5)

Before moving forward, let’s have a look at whether this dataset contains any null values or not: <p>

data.isnull().sum() <p>

![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/066c300c-9d74-48b6-82f3-e91b33bb643c)


The dataset is therefore ready to be used. After getting the first impressions of this data, I noticed the “smoker” column, which indicates whether the person smokes or not. This is an important feature of this dataset because a person who smokes is more likely to have major health problems compared to a person who does not smoke. So let’s look at the distribution of people who smoke and who do not: <p>

import plotly.express as px <p>
data = data  <p>
figure = px.histogram(data, x = "sex", color = "smoker", title= "Number of Smokers") <p>
figure.show() <p>
![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/365cb2ac-95a1-453d-bbe1-961f7a4e3ae0)

According to the above visualisation, 547 females, 517 males don’t smoke, and 115 females, 159 males do smoke. It is important to use this feature while training a machine learning model, so now I will replace the values of the “sex” and “smoker” columns with 0 and 1 as both these columns contain string values: <p>

data["sex"] = data["sex"].map({"female": 0, "male": 1})  <p>
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})  <p>
print(data.head())  <p>
![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/63e0fe66-5409-482b-96aa-7c68e53541f7)


Now let’s have a look at the distribution of the regions where people are living according to the dataset: <p>

pie = data["region"].value_counts()  <p>
regions = pie.index <p>
population = pie.values <p>
fig = px.pie(data, values=population, names=regions) <p>
fig.show()  <p>

![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/c286ccd3-c6cc-41b2-ba40-e85ded898250)

# Health Insurance Premium Prediction Model <p>
Now let’s move on to training a machine learning model for the task of predicting health insurance premiums. First, I’ll split the data into training and test sets <p>

x = np.array(data[["age", "sex", "bmi", "smoker"]]) <p>
y = np.array(data["charges"]) <p>

from sklearn.model_selection import train_test_split <p>
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42) <p>

After using different machine learning algorithms, I found the random forest algorithm as the best performing algorithm for this task. So here I will train the model by using the random forest regression algorithm: <p>

from sklearn.ensemble import RandomForestRegressor <p>
forest = RandomForestRegressor() <p>
forest.fit(xtrain, ytrain) <p>

**Now let’s have a look at the predicted values of the model:** <p>
ypred = forest.predict(xtest) <p>
data = pd.DataFrame(data={"Predicted Premium Amount": ypred}) <p>
print(data.head()) <p>

![image](https://github.com/KalyanKumarBhogi/Health_Insurance_Premium_Prediction/assets/144279085/530eccdd-bd6a-411c-be9b-353d3eae74da)

# Conclusion <p>
The health insurance premium prediction model, based on a dataset containing factors such as age, gender, BMI, smoking status, and region, utilized the random forest regression algorithm. The model successfully predicted premium amounts, taking into account the significant impact of variables like smoking status on health risks. The dataset preprocessing involved mapping categorical values, such as gender and smoking, to numerical representations. The analysis highlighted the distribution of smokers and non-smokers in the dataset. The trained model demonstrated its efficacy in predicting health insurance premiums, providing a valuable tool for insurers to assess personalized risk factors and set appropriate premium amounts for individuals. This approach contributes to a more accurate and data-driven estimation of health insurance costs based on individual characteristics. <p>
