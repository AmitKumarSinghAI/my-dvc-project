import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error
import pickle

data = pd.read_csv("data/processed.csv")


x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.2,random_state=2)

col = ColumnTransformer(
    [
        ("OHE", OneHotEncoder(drop="first", handle_unknown="ignore"), ["name"])
    ],
    remainder="passthrough"
)

x_train_ohe = col.fit_transform(x_train)
x_test_ohe  = col.transform(x_test)

lr = LinearRegression()
lr.fit(x_train_ohe,y_train)
pred = lr.predict(x_test_ohe)



with open("data/model.pkl", "wb") as file:
    pickle.dump(lr, file)