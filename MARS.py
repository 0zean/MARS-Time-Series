import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import talib
from pyearth import Earth
from scipy.stats import skew
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv('Data/AAPL5_train.csv')  # 5-minute Apple data
test = pd.read_csv('Data/AAPL5_test.csv')

# Preprocess: remove unnecessary columns
cols = list(train)[4:9]
train = train[cols].astype(str)

for i in cols:
    for j in range(0, len(train)):
        train[i][j] = train[i][j].replace(",", "")

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
x = train
del x['<HIGH>']
del x['<LOW>']
del x['<CLOSE>']
del x['<VOL>']

numeric_feats = x.dtypes[x.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda g: skew(g.dropna()))
skewed_feats = skewed_feats.index

x[skewed_feats] = np.log1p(x[skewed_feats])

# Fit MARS
mars = Earth(allow_missing=True)
mars.fit(x, y)
print(mars.trace())
print(mars.summary())


def inverse(x):
    x = np.exp(x) - 1
    return x


def graph(x, y, y2, a, b, Title):
    fig = plt.figure()
    plt.plot(x[a:b], y[a:b], 'r', label='Actual')
    plt.plot(x[a:b], y2[a:b], 'b', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(Title)
    plt.legend(loc='upper left')
    plt.show()
    return fig


# Predict training series
y_hat = mars.predict(x)
x_train = list(range(0, len(y)))

# Process test data
test = test[cols].astype(str)
for i in cols:
    for j in range(0, len(test)):
        test[i][j] = test[i][j].replace(",", "")

test = test.astype(float)

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
# x1 = test.drop('<CLOSE>',1)
x1 = test
del x1['<HIGH>']
del x1['<LOW>']
del x1['<CLOSE>']
del x1['<VOL>']

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

# Graphs of test/train
graph(x_train, inverse(y), inverse(y_hat), 5000, 5100,
      Title='MARS: Train').savefig('MARS1.png')

graph(x_test, inverse(y1), inverse(y_hat1), 150, 250,
      Title='MARS: Test').savefig('MARS2.png')

graph(x_train, inverse(y), inverse(yb), 5000, 5100,
      Title='Adaboost MARS: Train').savefig('AB1.png')

graph(x_test, inverse(y1), inverse(yb1), 150, 250,
      Title='Adaboost MARS: Test').savefig('AB2.png')

# Mean Squared Error on each case
MSE1 = mean_squared_error(inverse(y), inverse(y_hat))
MSE2 = mean_squared_error(inverse(y1), inverse(y_hat1))
MSEB1 = mean_squared_error(inverse(y), inverse(yb))
MSEB2 = mean_squared_error(inverse(y1), inverse(yb1))

# R-squared
R1 = r2_score(inverse(y), inverse(y_hat))
R2 = r2_score(inverse(y1), inverse(y_hat1))
R3 = r2_score(inverse(y), inverse(yb))
R4 = r2_score(inverse(y1), inverse(yb1))

# S-Statistics
S1 = pd.DataFrame(((pd.DataFrame(inverse(y)).values -
                   (pd.DataFrame(inverse(y_hat))).values)**2))

S11 = np.sqrt((S1.sum(axis=0))/len(y))

S2 = pd.DataFrame(((pd.DataFrame(inverse(y1)).values -
                   (pd.DataFrame(inverse(y_hat1))).values)**2))

S21 = np.sqrt((S2.sum(axis=0))/len(y1))

S3 = pd.DataFrame(((pd.DataFrame(inverse(y)).values -
                   (pd.DataFrame(inverse(yb))).values)**2))

S31 = np.sqrt((S3.sum(axis=0))/len(y))

S4 = pd.DataFrame(((pd.DataFrame(inverse(y1)).values -
                   (pd.DataFrame(inverse(yb1))).values)**2))

S41 = np.sqrt((S4.sum(axis=0))/len(y1))


plotly.offline.plot({
    "data": [go.Table(
        header=dict(values=['Type', 'MSE', 'R-Squared', 'SE of Estimate'],
                    align=['left']*5),
        cells=dict(values=[['train', 'OOS', 'train(boosted)', 'OOS(boosted)'],
                   [MSE1, MSE2, MSEB1, MSEB2],
                   [R1, R2, R3, R4],
                   [S11, S21, S31, S41]], align=['left']*5))]},
                   image_filename='stats', image='png',
                   image_height=400, image_width=300)
