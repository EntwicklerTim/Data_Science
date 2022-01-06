import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import create_engine
import datetime
date = datetime.datetime.now()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from insert_data.tech_ingest import TechInsert
from sklearn.svm import SVC

class Predicition:
    def __init__(self,):
        self.tech_insert = TechInsert
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')

    def predict_tech(self):
        # The best Mlmodel and its parameters are assigned to the respective cryptocurrencies
        differen_dict = {
            "bitcoin": SVC(C=  0.2, kernel =  'linear'),
            "ethereum":  SVC(),
            "binancecoin":  DecisionTreeClassifier(max_depth = 1),
            "cardano":  RandomForestClassifier(max_depth=  6, n_estimators = 10),
            "litecoin":  KNeighborsClassifier(n_neighbors = 10),
            "ripple":  RandomForestClassifier(max_depth=  2, n_estimators = 20),
            "monero": KNeighborsClassifier(n_neighbors = 19),
            "tron":  KNeighborsClassifier(n_neighbors = 19),
            "bitcoin-cash":  DecisionTreeClassifier()
        }
        #load prices
        price = pd.read_sql("SELECT * FROM cryptogroup2.HistCoins", self.engine_ds.connect())
        pivot_price = pd.pivot_table(price, index="date", columns="id", values="price")
        pivot_price.fillna(method="ffill", inplace=True)
        all_id = list(price["id"].unique())
        price_Chagne_DF = pd.DataFrame()
        pivot_price_BHS = {}
        #load technical indicators
        tech_table = pd.read_sql("SELECT * FROM cryptogroup2.tech_indikator", self.engine_ds.connect())
        tech_dict_full = {}
        tech_dict_train ={}
        prediction_target = {}
        #Converting the technical indicators into the desired table of characteristics
        for x in all_id:
            tech_dict_full[x] = tech_table[tech_table["asset"] == x]
            tech_dict_full[x] = tech_dict_full[x].apply(lambda x: self.transform(x), axis=1)
            index = tech_table[tech_table["asset"] == x].date
            tech_dict_full[x] = pd.DataFrame(
                [[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p in
                 tech_dict_full[x].values],
                columns=["bollinger_signal", "rsi_signal", "df_MACD", "df_MACD_signal", "df_KAMA", "PPO_hist",
                         "PPO_Signal", "df_STOCHrsi", "df_TSI", "df_FI", "df_OBV", "df_ULI", "df_Aroon", "df_kst",
                         "df_STC", "df_PVO"])
            tech_dict_full[x].set_index(index, inplace=True)
            tech_dict_train[x] = tech_dict_full[x][:-31]# feature


        #transform price table
        for x in all_id:
            price_Chagne_DF[x] = price[price["id"] == x].set_index("date")["price"].pct_change(30).dropna()[72:]
            pivot_price_BHS[x] = price_Chagne_DF[x].apply(lambda x: 1 if (x > 0.05) else 0 if (x <= 0.05 and x >= -0.05) else -1)#target


        #Train the model and predict price direction
        for x in differen_dict:
            prediction_model = differen_dict[x]
            print(prediction_model)
            feature = tech_dict_train[x]
            target = pivot_price_BHS[x]
            feature_train = feature[:round(len(feature)*0.90)]
            target_train = target[:round(len(feature)*0.90)]
            predicition_feature = tech_dict_full[x][-30:]
            prediction_model.fit(feature_train, target_train)
            prediction_target[x] = prediction_model.predict(predicition_feature)
        # persist the price direction for the thirtieth day in the future for each cryptocurrency
        pd.DataFrame(prediction_target)[-1:].transpose().reset_index().to_sql("prediction_signal", con=self.engine_ds, if_exists='replace', index=False, chunksize=1000)

    def transform(self, value):
        # Converting the value of the technical indicator into discrete values, according to the respective interpretation of the indicator
        bollinger_signal = 1 if (value["df_bollinger_cross_low"] == 1) else  -1 if (value["df_bollinger_cross_high"] == 1) else  0
        rsi_signal = 1 if (value["rsi"] <= 30) else -1 if (value["rsi"] >= 70) else 0
        macd_signal = 1 if (value["df_MACD"] > 0) else -1 if (value["df_MACD"] < 0) else 0
        df_MACD_signal = 1 if (value["df_MACD_signal"] > 0) else -1 if(value["df_MACD_signal"] < 0) else 0
        df_KAMA = 1 if (value["df_KAMA"] > 0.05) else -1 if (value["df_KAMA"] < -0.05) else 0
        df_PPO_hist = 1 if (value["df_PPO_hist"] > 0) else -1 if (value["df_PPO_hist"] < 0) else 0
        df_PPO_signal = 1 if (value["df_PPO_signal"] > 0) else -1 if (value["df_PPO_signal"] < 0) else 0
        df_STOCHrsi = 1 if (value["df_STOCHrsi"] <= 0.20) else -1 if (value["df_STOCHrsi"] >= 0.80) else 0
        df_TSI = 1 if (value["df_TSI"] > 0) else -1 if (value["df_TSI"] < 0) else 0
        df_FI = 1 if (value["df_FI"] > 0) else -1 if (value["df_FI"] < 0) else 0
        df_OBV = 1 if (value["df_OBV"] > 0.02) else -1 if (value["df_OBV"] < -0.02) else 0
        df_ULI = 1 if (value["df_ULI"] > 12) else 0
        df_Aroon = 1 if (value["df_Aroon"] > 30) else -1 if (value["df_Aroon"] < -30) else 0
        df_kst = 1 if (value["df_kst"] > 0) else -1 if (value["df_kst"] < 0) else 0
        df_STC = 1 if (value["df_STC"] > 0) else -1 if (value["df_STC"] < 0) else 0
        df_PVO = 1 if (value["df_PVO"] > 0) else -1 if (value["df_PVO"] < 0) else 0

        return (bollinger_signal, rsi_signal, macd_signal, df_MACD_signal, df_KAMA, df_PPO_hist,
                df_PPO_signal, df_STOCHrsi, df_TSI, df_FI, df_OBV, df_ULI, df_Aroon, df_kst, df_STC, df_PVO)