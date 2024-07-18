# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings("ignore")
# In[2]:
df = pd.read_csv("dataset/cardekho.csv")
# In[3]:
df.shape
# In[4]:
df.describe()
# In[5]:
df.head()
# In[6]:
df.info()
# In[7]:
df.isna().sum()
# In[8]:
df.rename(columns={'mileage(km/ltr/kg)': 'mileage'}, inplace=True)
# In[9]:
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
# In[10]:
# Impute missing values with mean for numeric columns
numeric_cols = ['mileage', 'engine', 'max_power', 'seats']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Confirm the NaN values are filled
print(df.isna().sum())
# In[11]:
df.nunique()
# # EDA
# In[12]:
columns_to_include = ['year', 'selling_price', 'km_driven',
                      'fuel', 'mileage', 'engine', 'max_power', 'seats']
sns.set(style="whitegrid")
sns.pairplot(df[columns_to_include], hue='selling_price',
             palette="viridis", height=2.5)
plt.suptitle("Pair Plot of Car Data", y=1.02, fontsize=16)
plt.show()
# In[13]:
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
sns.countplot(x='fuel', data=df, hue='fuel',
              palette='Set2', ax=axes[0, 0], legend=False)
axes[0, 0].set_title('Count of Cars by Fuel Type')
sns.countplot(x='seller_type', data=df, hue='seller_type',
              palette='Set2', ax=axes[0, 1], legend=False)
axes[0, 1].set_title('Count of Cars by Seller Type')

sns.countplot(x='transmission', data=df, hue='transmission',
              palette='Set2', ax=axes[1, 0], legend=False)
axes[1, 0].set_title('Count of Cars by Transmission Type')

sns.countplot(x='owner', data=df, hue='owner',
              palette='Set2', ax=axes[1, 1], legend=False)
axes[1, 1].set_title('Count of Cars by Owner')
plt.xticks(rotation=45)
plt.tight_layout(pad=1.0)  # Adjust the padding between plots
plt.show()
# In[14]:

plt.figure(figsize=(12, 6))
sns.barplot(x='year', y='selling_price', data=df, palette='viridis')
plt.title('Average Selling Price by Car Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Selling Price', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[15]:
df.head()


# In[16]:
df_sorted = df.sort_values(by='selling_price', ascending=False)
top_10_df = df_sorted.head(10)

car_names = top_10_df['name']
selling_prices = top_10_df['selling_price']

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(car_names))

bar1 = ax.bar(index, selling_prices, bar_width, label='Selling Price')

ax.set_xlabel('Car Names')
ax.set_ylabel('Price ')
ax.set_title('Top 10 Cars Based on Selling Price')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(car_names, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()
# In[17]:

label_encoders = {
    'name': LabelEncoder(),
    'fuel': LabelEncoder(),
    'seller_type': LabelEncoder(),
    'transmission': LabelEncoder(),
    'owner': LabelEncoder()
}
df['name'] = label_encoders['name'].fit_transform(df['name'])
df['fuel'] = label_encoders['fuel'].fit_transform(df['fuel'])
df['seller_type'] = label_encoders['seller_type'].fit_transform(
    df['seller_type'])
df['transmission'] = label_encoders['transmission'].fit_transform(
    df['transmission'])
df['owner'] = label_encoders['owner'].fit_transform(df['owner'])

# In[18]:

# Features and target
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']
# In[19]:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# # Random Forest Regressor

# In[20]:

# Training the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# In[21]:
# Training the LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
# In[22]:
# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)
# In[23]:
# Evaluation
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f'Mean Squared Error: {rf_mse:.2f}')
print(f'R-squared: {rf_r2:.2f}')

# In[24]:

lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f'Mean Squared Error: {lr_mse:.2f}')
print(f'R-squared: {lr_r2:.2f}')

# In[25]:
# Compare and save the best model
if rf_r2 > lr_r2:
    best_model = rf_model
    model_name = "Random Forest"
else:
    best_model = lr_model
    model_name = "Linear Regression"
# In[26]:

with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
# In[27]:

print(
    f"The best model is {model_name} and has been saved as 'best_model.pkl'.")


# In[28]:
# Plotting feature importance for Random Forest (only if it is the best model)
if model_name == "Random Forest":
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(feature_importance)[::-1]

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance[indices], y=[
                feature_names[i] for i in indices])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
# In[29]:
with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)
