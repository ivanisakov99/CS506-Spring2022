from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn.linear_model import ElasticNet, LinearRegression, ridge_regression
import time
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
# ans 17.34 something
df = pd.read_csv('data.csv')

def convert(date):
    return time.mktime(time.strptime(date, "%H:%M:%S"))

df['alarm (%H:%M:%S)'] = df['alarm (%H:%M:%S)'].apply(convert)

model = sm.OLS(df['alarm (%H:%M:%S)'].values, sm.add_constant(df['day'].values))
results = model.fit()
print(model.predict([365]))



# # model = ElasticNet().fit(df['day'].values.reshape(-1, 1),
# #                                df['alarm (%H:%M:%S)'].values.reshape(-1, 1))

# ans = model.predict(np.array([[356]]))
# print(ans)
# # print(datetime.fromtimestamp(ans[0][0], tz=pytz.timezone('US/Eastern')))
# print(datetime.fromtimestamp(ans[0][0]))
