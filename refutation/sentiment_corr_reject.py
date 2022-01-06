from sqlalchemy import create_engine
import datetime
from datetime import datetime
import pandas as pd
from insert_data.tech_ingest import  TechInsert

class Refute:
    def __init__(self):
        self.load_crypto = TechInsert()
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')

    def refute(self):
        sentiment = pd.read_sql("SELECT date, CryptoID, Avg_Polarity, Dep_Polarity  FROM cryptogroup2.Tweets_Grouped", self.engine_ds.connect())
        sentiment["date"] = pd.to_datetime(sentiment['date']).dt.date
        senti_pivot = pd.pivot_table(sentiment, index="date",columns="CryptoID", aggfunc='median' ).rename(columns={"BTC":"bitcoin","BNB":"binancecoin","BCH":"bitcoin-cash", "ADA":"cardano","ETH":"ethereum","LTC":"litecoin","XRP":"ripple", "TRX": "tron", "XMR":"monero", "SOL":"solana"}).fillna(0)
        senti_table = senti_pivot[:-30]
        crypto_table = self.load_crypto.load().iloc[self.load_crypto.load().index.searchsorted(datetime(2018, 11, 1)):]
        corr_dict_avg_pol = {}
        corr_dict_dep_pol = {}
        corr_dict = {}
        for x in list(self.load_crypto.load().columns):
            corr_dict_avg_pol[x] = senti_table["Avg_Polarity"][x].corr(crypto_table[x])
            corr_dict_dep_pol[x] = senti_table["Dep_Polarity"][x].corr(crypto_table[x])
            corr_dict[x] = (senti_table["Avg_Polarity"][x].corr(crypto_table[x]),senti_table["Dep_Polarity"][x].corr(crypto_table[x]))

        frame_index = pd.DataFrame(corr_dict).pivot_table(pd.DataFrame(corr_dict), columns = [0,1]).rename(columns = {0: 'Avg_Polarity', 1:'Dep_Polarity'})
        frame_index.reset_index(level=0, inplace=True)
        frame_index.to_sql("view_reject_sentiment", con=self.engine_ds, if_exists='replace', index=False, chunksize=1000)
