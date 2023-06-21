import numpy as np
import pandas as pd
import talib
from pyearth import Earth
from scipy.stats import skew
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

from plot_utils import graph, table

train = pd.read_csv('Data/AAPL5_train.csv')  # 5-minute Apple data
test = pd.read_csv('Data/AAPL5_test.csv')

# Preprocess: remove unnecessary columns
cols = list(train)[4:9]
train = train[cols]

train = train.astype(float)

HT = talib.HT_DCPERIOD(train['<OPEN>'])
std = talib.STDDEV(train['<OPEN>'], timeperiod=7, nbdev=1)

HT = pd.DataFrame(data={'HT_DCPERIOD': HT})
std = pd.DataFrame(data={'STDDEV': std})

train = train.join(HT)
train = train.join(std)

avgHT1 = train['HT_DCPERIOD'].mean()
avgSTD = train["STDDEV"].mean()

train['HT_DCPERIOD'].fillna(avgHT1, inplace=True)
train['STDDEV'].fillna(avgSTD, inplace=True)

# transform data using log(1+x)
train['<HIGH>'] = np.log1p(train['<HIGH>'])
y = train['<HIGH>']
x = train[['<OPEN>', 'HT_DCPERIOD', 'STDDEV']]

numeric_feats = x.dtypes[x.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda g: skew(g.dropna()))
skewed_feats = skewed_feats.index

x[skewed_feats] = np.log1p(x[skewed_feats])

# Fit MARS
mars = Earth(allow_missing=True, thresh=0.00001, max_degree=3)
mars.fit(x, y)
print(mars.trace())
print(mars.summary())

# Predict training series
y_hat = mars.predict(x)
x_train = list(range(0, len(y)))

# Process test data
test = test[cols]
train = train.astype(float)

HT = talib.HT_DCPERIOD(test['<OPEN>'])
std = talib.STDDEV(test['<OPEN>'], timeperiod=7, nbdev=1)

HT = pd.DataFrame(data={'HT_DCPERIOD': HT})
std = pd.DataFrame(data={'STDDEV': std})

test = test.join(HT)
test = test.join(std)

avgHT1 = test['HT_DCPERIOD'].mean()
avgSTD = test["STDDEV"].mean()

test['HT_DCPERIOD'].fillna(avgHT1, inplace=True)
test['STDDEV'].fillna(avgSTD, inplace=True)

test['<HIGH>'] = np.log1p(test['<HIGH>'])
y1 = test['<HIGH>']
x1 = test[['<OPEN>', 'HT_DCPERIOD', 'STDDEV']]

features = x1.dtypes[x1.dtypes != "object"].index
features_skewed = test[features].apply(lambda g: skew(g.dropna()))
features_skewed = features_skewed.index
x1[features_skewed] = np.log1p(x1[features_skewed])

# Predict Test series
y_hat1 = mars.predict(x1)
x_test = list(range(0, len(y1)))

# Adaboost MARS
boosted_mars = AdaBoostRegressor(estimator=mars,
                                 n_estimators=50,
                                 learning_rate=0.1,
                                 loss="exponential")
boosted_mars.fit(x, y)

# Predict using boosted MARS
yb = boosted_mars.predict(x)
yb1 = boosted_mars.predict(x1)


# Inverse log(1+p) transform
def inverse(x):
    x = np.exp(x) - 1
    return x


y, y_hat = inverse(y), inverse(y_hat)
y1, y_hat1 = inverse(y1), inverse(y_hat1)
yb, yb1 = inverse(yb), inverse(yb1)

# Graphs of test/train
graph(x_train, y, y_hat, 5000, 5100,
      Title='MARS: Train').savefig('Plots/MARS1.png')

graph(x_test, y1, y_hat1, 150, 250,
      Title='MARS: Test').savefig('Plots/MARS2.png')

graph(x_train, y, yb, 5000, 5100,
      Title='Adaboost MARS: Train').savefig('Plots/AB1.png')

graph(x_test, y1, yb1, 150, 250,
      Title='Adaboost MARS: Test').savefig('Plots/AB2.png')


# Mean Squared Error on each case
MSE1 = mean_squared_error(y, y_hat)
MSE2 = mean_squared_error(y1, y_hat1)
MSEB1 = mean_squared_error(y, yb)
MSEB2 = mean_squared_error(y1, yb1)

# R-squared
R1 = r2_score(y, y_hat)
R2 = r2_score(y1, y_hat1)
R3 = r2_score(y, yb)
R4 = r2_score(y1, yb1)


# S-Statistic
def s_error(true, pred):
    s = pd.DataFrame(((pd.DataFrame(true).values -
                     (pd.DataFrame(pred)).values)**2))
    se = np.sqrt((s.sum(axis=0))/len(true))
    return se[0]


S1 = s_error(y, y_hat)
S2 = s_error(y1, y_hat1)
S3 = s_error(y, yb)
S4 = s_error(y1, yb1)

# Plot Table of Metrics
table_data = {
    "": ["Train", "OOS", "Train (Boosted)", "OOS (Boosted)"],
    "MSE": [f'{MSE1:.3f}', f'{MSE2:.3f}', f'{MSEB2:.3f}', f'{MSEB2:.3f}'],
    "R\u00b2": [f'{R1:.4f}', f'{R2:.4f}', f'{R3:.4f}', f'{R4:.4f}'],
    "Standard Error": [f'{S1:.3f}', f'{S2:.3f}', f'{S3:.3f}', f'{S4:.3f}']
}

table(pd.DataFrame(table_data)).savefig('Plots/stats.png')
