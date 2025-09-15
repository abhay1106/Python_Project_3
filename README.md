## COVID-19 Data Analysis and Forecasting

## Problem Statement
Given data about COVID-19 patients, this project visualizes the impact and analyzes the trend of infection and recovery rates. It also predicts the number of cases expected a week into the future based on current trends.

## Dataset
- CSV file: `covid_19.csv`
- Contains data about confirmed, recovered, and death cases worldwide and in India.

## Solution
The solution involves the following steps:

### 1. Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
import warnings
```

### 2. Load and Inspect Dataset
```python
df = pd.read_csv('covid_19.csv')
df.rename(columns={
    "Province/State":"state", "Country/Region":"country", "Lat":"latitude",
    "Long":"longitude", "Date":"date", "Confirmed":"confirmed",
    "Deaths":"deaths", "Recovered":"recovered", "Active":"active",
    "WHO Region":"Region"
}, inplace=True)
```

### 3. Exploratory Data Analysis (EDA)
### **Compute active cases:**
```python
df["active"] = df["confirmed"] - df["deaths"] - df["recovered"]
```
### **Group data by country:**
```python
grouped_df = df.groupby("country")[["confirmed","active","recovered","deaths"]].sum().reset_index()
```
## 4. Visualizations
### **4.1 Global Cases**

Active cases:
```python
fig = px.choropleth(grouped_df, locations="country", locationmode="country names",
                    color="active", hover_name="country", range_color=[1,1500],
                    color_continuous_scale="reds", title="Active Cases")
fig.show()
```

Recovered cases:
```python
fig1 = px.choropleth(grouped_df, locations="country", locationmode="country names",
                     color="recovered", hover_name="country", range_color=[1,2500],
                     color_continuous_scale="greens", title="Recovered Cases")
fig1.show()
```
Death cases:
```python
fig2 = px.choropleth(grouped_df, locations="country", locationmode="country names",
                     color="deaths", hover_name="country", range_color=[1,4500],
                     color_continuous_scale="blackbody", title="Death Cases")
fig2.show()
```

### **4.2 Trends Over Time**

Confirmed cases trend:
```python
confirmed_cases = df.groupby("date")["confirmed"].sum().reset_index()
plt.figure(figsize=(15,10))
sns.pointplot(x=pd.to_datetime(confirmed_cases.date).dt.date, y=confirmed_cases.confirmed, color="r")
plt.xticks(rotation=90, fontsize=5)
plt.show()
```
## 4.3 Top 20 Countries

Top 20 Countries by Active Cases:
```python
top_actives = df.groupby("country")["active"].sum().sort_values(ascending=False).head(20).reset_index()

plt.figure(figsize=(15,10))
plt.title("Top 20 Countries having most Active Cases", fontsize=30)
fig = sns.barplot(x=top_actives.active, y=top_actives.country)
for i,(value,name) in enumerate(zip(top_actives.active, top_actives.country)):
    fig.text(value, i-0.005, f"{value:,.0f}", size=10, ha="left", va="center")
plt.xlabel("Total Active Cases (In Millions)", fontsize=20)
plt.ylabel("Country", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

Top 20 Countries by Death Cases:
```python
top_deaths = df.groupby("country")["deaths"].sum().sort_values(ascending=False).head(20).reset_index()

plt.figure(figsize=(15,10))
plt.title("Top 20 Countries having most Death Cases", fontsize=30)
fig = sns.barplot(x=top_deaths.deaths, y=top_deaths.country)
for i,(value,name) in enumerate(zip(top_deaths.deaths, top_deaths.country)):
    fig.text(value, i-0.005, f"{value:,.0f}", size=10, ha="left", va="center")
plt.xlabel("Total Death Cases", fontsize=20)
plt.ylabel("Country", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
```
Top 20 Countries by Recovered Cases:
```python
top_recovered = df.groupby("country")["recovered"].sum().sort_values(ascending=False).head(20).reset_index()

plt.figure(figsize=(15,10))
plt.title("Top 20 Countries having most Recovered Cases", fontsize=30)
fig = sns.barplot(x=top_recovered.recovered, y=top_recovered.country)
for i,(value,name) in enumerate(zip(top_recovered.recovered, top_recovered.country)):
    fig.text(value, i, f"{value:,.0f}", size=10, ha="left", va="center")
plt.xlabel("Total Recovered Cases", fontsize=20)
plt.ylabel("Country", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
```
## 5. Country-wise Analysis
Brazil, US, Russia, India, Germany:
```python
brazil = df[df.country=="Brazil"].groupby("date")[["recovered","deaths","confirmed","active"]].sum().reset_index()
# Repeat for US, Russia, India, Germany
```

### **6. Total Cases Summary**
```python
total_active = df["active"].sum()
total_confirmed = df["confirmed"].sum()
total_recovered = df["recovered"].sum()
total_deaths = df["deaths"].sum()
print(total_active, total_confirmed, total_recovered, total_deaths)
```

## 7. Time Series Forecasting with Prophet
### **7.1 Forecasting Confirmed Cases**
```python
confirmed = df.groupby("date").sum()["confirmed"].reset_index()
confirmed.columns = ["ds","y"]
confirmed["ds"] = pd.to_datetime(confirmed["ds"])

model = Prophet(interval_width=0.95)
model.fit(confirmed)

future = model.make_future_dataframe(periods=7, freq="D")
forecast = model.predict(future)
model.plot(forecast)
```

### **7.2 Forecasting Recovered Cases**
```python
recovered = df.groupby("date").sum()["recovered"].reset_index()
recovered.columns = ["ds","y"]
recovered["ds"] = pd.to_datetime(recovered["ds"])

model1 = Prophet(interval_width=0.95)
model1.fit(recovered)

future1 = model1.make_future_dataframe(periods=7, freq="D")
forecast1 = model1.predict(future1)
model1.plot(forecast1)
```

### **7.3 Forecasting Death Cases**
```python
deaths = df.groupby("date").sum()["deaths"].reset_index()
deaths.columns = ["ds","y"]
deaths["ds"] = pd.to_datetime(deaths["ds"])

model2 = Prophet(interval_width=0.95)
model2.fit(deaths)

future2 = model2.make_future_dataframe(periods=7, freq="D")
forecast2 = model2.predict(future2)
model2.plot(forecast2)
```
## 8. Results

- Total Active, Confirmed, Recovered, and Death cases worldwide.

- Top 20 countries visualizations.

- Trends of confirmed cases over time for major countries.

- 7-day forecasts for confirmed, recovered, and death cases.

## 9.Key Learnings

- Pandas and Plotly simplify data analysis and visualization.

- Active cases calculation highlights the current pandemic status.

- Prophet enables short-term forecasting of time-series data.

- Visualizations reveal global and regional COVID-19 trends.

## 10. Conclusion

The analysis provides insights into COVID-19 spread and recovery patterns worldwide and in India. Forecasts help plan and respond effectively. Combining EDA, visualizations, and forecasting demonstrates the power of data-driven decision-making.
