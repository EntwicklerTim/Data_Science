import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.svm import SVC
#import pandas_datareader as web
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import datetime
from datetime import timedelta
#from coinmetrics.api_client import CoinMetricsClient
#import coinmetrics
import ta

date = datetime.datetime.now()
gestern = pd.Timestamp(datetime.datetime.now().date() - timedelta(days=1))
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from insert_data.tech_ingest import TechInsert

class Predicition:
    def __init__(self,):
        self.tech_insert = TechInsert
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')



    def predict_tech(self):

        # transform techtable
        tech_table = pd.read_sql("SELECT * FROM cryptogroup2.tech_indikators", self.engine_ds.connect()) #tech_table

        tech_dict = {}
        for x in list(tech_table.asset.unique()):
            tech_dict[x] = tech_table[tech_table["asset"] == x]
            tech_dict[x] = tech_dict[x].apply(lambda x: self.transform(x), axis=1)
            index = tech_table[tech_table["asset"] == x].date
            tech_dict[x] = pd.DataFrame([[a, b, c, d] for a, b, c, d in tech_dict[x].values],
                                        columns=["bollinger_signal", "rsi_signal", "macd_signal",
                                                 "df_ema9_macd_hist_signal"])
            tech_dict[x].set_index(index, inplace=True)
            tech_dict[x] = tech_dict[x][:-28]# feature
        del tech_dict["solana"]
        #return tech_dict


        #transform price table
        price_table = self.tech_insert.load(self)#.dropna(axis=1,inplace=True)
        price_table.dropna(axis=1, inplace=True)
        pivot_price_change = price_table.pct_change(30)
        pivot_price_change.dropna(axis=0, inplace=True)
        pivot_price_BHS = pivot_price_change.applymap(lambda x: 1 if (x > 0.10) else 0 if (x <= 0.05 and x >= -0.05) else -1) # alternativ  just buy and sell
        pivot_price_BHS_shift = pivot_price_BHS[31:] #target
        print(pivot_price_BHS_shift)

        prediction_dict = {}
        predition_feature = {}
        for x in list(pivot_price_BHS_shift.columns):
            prediction_dict[x] = DecisionTreeClassifier()
            feature = tech_dict[x]
            print(feature)
            target = pivot_price_BHS_shift[x]
            #feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.10)
            feature_train = np.array(feature[:1000]).reshape(-1, 1)
            target_train = np.array(feature[:1000]).reshape(-1, 1)
            feature_test = np.array(target[-100:]).reshape(-1, 1)
            target_test = np.array(target[-100:]).reshape(-1, 1)
            feature_train = feature[:1200]
            target_train = target[:1200]
            feature_test = feature[-100:]
            target_test = target[-100:]
            predicition_feature = feature[-30:]
            prediction_dict[x].fit(feature_train, target_train)
            score = prediction_dict[x].score(feature_test, target_test)
            predition_feature[x] = prediction_dict[x].predict(predicition_feature)

        prediction_recomm_30_day = pd.DataFrame(predition_feature)[-1:].transpose().reset_index()
        print(prediction_recomm_30_day)
        prediction_recomm_30_day.to_sql("prediction_signal", con=self.engine_ds, if_exists='replace', index=False, chunksize=1000)


    def transform(self, value):
        if value["Bollinger_cross_low"] == 1:
            bollinger_signal = 1
        elif value["Bollinger_cross_high"] == 1:
            bollinger_signal = -1
        else:
            bollinger_signal = 0

        if value["rsi"] <= 30:
            rsi_signal = 1
        elif value["rsi"] >= 70:
            rsi_signal = -1
        else:
            rsi_signal = 0

        if value["macd"] > 0:
            macd_signal = 1
        elif value["macd"] < 0:
            macd_signal = -1
        else:
            macd_signal = 0

        if value["df_ema9_macd_hist"] > 0:
            df_ema9_macd_hist_signal = 1
        elif value["df_ema9_macd_hist"] > 0:
            df_ema9_macd_hist_signal = -1
        else:
            df_ema9_macd_hist_signal = 0

        return (bollinger_signal, rsi_signal, macd_signal, df_ema9_macd_hist_signal)


    """def predict_metric_price(self):
        #get the metrics
        metric_indikator = pd.read_sql("SELECT * FROM cryptogroup2.CoinMetric", self.engine_ds.connect())
        range_liste = list(filter(None, metric_indikator.asset.unique()))
        metric_tab_dict = {}
        metric_tabelle_dict_shifted = {}
        for x in range_liste:
            metric_tabelle = metric_indikator[metric_indikator["asset"] == x]
            #print(metric_tabelle)
            metric_tab_dict[x] = metric_tabelle
            metric_tab_dict[x] = pd.pivot_table(metric_tab_dict[x], index="date", columns="metric_name")
            metric_tab_dict[x].columns = metric_tab_dict[x].columns.droplevel()
            print(metric_tab_dict[x])
            metric_tabelle_dict_shifted[x] = metric_tab_dict[x][:-29]


        # get the prices
        price = pd.read_sql("SELECT * FROM cryptogroup2.HistCoins", self.engine_ds.connect())
        pivot_price = pd.pivot_table(price, index="date", columns="id", values="price")
        pivot_price.fillna(method="ffill", inplace=True)
        pivot_price.dropna(axis=1, inplace=True)

        prediction_dict_REG = {}
        prediction_dict_REG_FUT  = {}
        for x in range_liste:
            feature_train = metric_tabelle_dict_shifted[x][:1200] #metric_tabelle_dict_not_all
            target_train = pivot_price[x][:1200]
            feature_test = metric_tabelle_dict_shifted[x][-50:]
            target_test = pivot_price[x][-50:]
            predicition_feature = metric_tab_dict[x][-30:]

            #print(x)
            #print(predicition_feature)
            #print(feature_train)
            #print(target_train)
            #prediction_dict_metri_sgd[x] = SGDRegressor(max_iter=100)
            prediction_dict_REG[x] = make_pipeline(#PolynomialFeatures(1),
                                                   LinearRegression()
                )



            prediction_dict_REG[x].fit(feature_train, target_train)
            prediction_dict_REG_FUT[x] = prediction_dict_REG[x].predict(predicition_feature)"""

    def predict_metric_price(self):

        metric_tab_full = {}
        metric_tab_train = {}
        prediction_days = 60
        future_day_start = 30
        future_day_end = 31
        future_day = 27
        train_dict = {}
        scaler_dict_m = {}
        scaler_dict_p = {}
        metric_tab_train_tra = {}
        model_dict = {}
        model_dict_fit = {}
        metric_tabelle_dict_predict = {}
        test_dict = {}
        metric_tabelle_dict_predict_trans = {}
        prediction_dict = {}
        prediction_prices_back_trans = {}
        #prepare price
        price = pd.read_sql("SELECT * FROM cryptogroup2.HistCoins", self.engine_ds.connect())
        pivot_price = pd.pivot_table(price, index="date", columns="id", values="price")
        pivot_price.fillna(method="ffill", inplace=True)
        pivot_price.dropna(axis=1, inplace=True)


        #prepare metrics
        metric_indikator = pd.read_sql("SELECT * FROM cryptogroup2.CoinMetric", self.engine_ds.connect())
        range_liste = list(filter(None, metric_indikator.asset.unique()))
        for x in range_liste:
            print(pivot_price[x])
            print(x)
            metric_tab_full[x] = metric_indikator[metric_indikator["asset"] == x]
            metric_tab_full[x] = pd.pivot_table(metric_tab_full[x], index="date", columns="metric_name")
            metric_tab_full[x].columns = metric_tab_full[x].columns.droplevel() #until last day e.g whole dataset
            metric_tab_train[x] = metric_tab_full[x][:-28] #  subtraction of the last 28 Datapoints these 28 dataset we need for the prediction
            scaler_dict_m[x] = StandardScaler() #scaler for the metrics
            scaler_dict_p[x] = StandardScaler() # scaler for the price
            # transfer metric table
            metric_tab_train_tra[x] = scaler_dict_m[x].fit_transform(metric_tab_train[x])
            shape = len(list(metric_tab_full[x].columns)) #shape how much different metrics go into RNN
            # transfer price table
            price_tabel_trans = scaler_dict_p[x].fit_transform(np.array(pivot_price[x]).reshape(-1, 1))
            x_liste, y_liste = [], []

            for x1 in range(prediction_days, len(pivot_price[x]) - future_day): #  alternativ metric_tabelle_dict_not_all_trans, price meistens kürzer
                x_liste.append((metric_tab_train_tra[x][x1 - prediction_days:x1]).reshape(-1, shape))
                y_liste.append(float(price_tabel_trans[x1 + future_day - 1]))

            arr_liste_x, arr_liste_y = np.array(x_liste), np.array(y_liste).reshape(-1, 1)
            #arr_liste_x = np.array(x_liste)
            train_dict[x] = [arr_liste_x, arr_liste_y]
            print(train_dict)

        for x in range_liste:
            x_train = train_dict[x][0]
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
            # Dropoutlayer with 20% Dropout to omitt random 20% of the input to prevent overfitting.
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model_dict[x] = model

        for x in range_liste:
            x_train = train_dict[x][0]
            y_train = train_dict[x][1]
            # print(model_dict[x])
            model_dict[x].compile(optimizer='adam', loss='MeanAbsoluteError')
            model_dict[x].fit(x_train, y_train, epochs=20, batch_size=32)

            model_dict_fit[x] = model_dict[x]


        for x in range_liste:
            metric_tabelle_dict_predict[x] = metric_tab_full[x][-90:]
            metric_tabelle_dict_predict_trans[x] = scaler_dict_m[x].fit_transform(metric_tabelle_dict_predict[x])
            shape = len(list(metric_tabelle_dict_predict[x].columns))
            x_liste_p = []
            for x1 in range(prediction_days, len(metric_tabelle_dict_predict_trans[x])):
                x_liste_p.append((metric_tabelle_dict_predict_trans[x][x1 - prediction_days:x1]).reshape(-1, shape))

                # y_liste.append(float(price_tabel[x1 + future_day_start]))

            test_dict[x] = np.array(x_liste_p)

        for x in range_liste:
            prediction_dict[x] = model_dict_fit[x].predict(test_dict[x])
            prediction_prices_back_trans[x] = scaler_dict_p[x].inverse_transform(prediction_dict[x])
            print(prediction_prices_back_trans)





        #creation of the prediction Frame
        predicition_list = []
        for x1 in range_liste:
            for x2 in range(len(prediction_prices_back_trans[x1][:, 0])):
                predicition_list.append([x1, pd.Timestamp(datetime.datetime.now().date() + timedelta(days=x2 + future_day)),
                                         prediction_prices_back_trans[x1][x2, 0]])

        predictionFrame = pd.DataFrame(predicition_list, columns=["Asset", "Date", "Price"])
        predictionFrame.to_sql("prediction", con=self.engine_ds, if_exists='replace', index=False, chunksize=1000)