from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import datetime
from datetime import timedelta
import ta


class TechInsert:
    def __init__(self,):
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')


    def load(self):
        df_db = pd.read_sql("SELECT * FROM cryptogroup2.HistCoins", self.engine_ds.connect())
        pivot_df = pd.pivot_table(df_db, index="date", columns="id", values="price")
        return pivot_df

    def calculate_tech_ind(self, data):
        df_RSI = pd.DataFrame()
        for x in data.columns:
           rsi_21 = ta.momentum.RSIIndicator(data[x], window=21)
           df_RSI[x] = rsi_21.rsi()

        df_MACD = pd.DataFrame()
        for x in data.columns:
            macd_16 = ta.trend.macd(data[x], window_slow=26, window_fast=12)
            df_MACD[x] = macd_16

        df_ema9_macd = pd.DataFrame()
        for x in data.columns:
            ema9 = ta.trend.EMAIndicator(df_MACD[x], window=9)
            df_ema9_macd[x] = ema9.ema_indicator()

        df_ema9_macd_hist = pd.DataFrame()
        for x in data.columns:
            df_ema9_macd_hist[x] = df_ema9_macd[x] - df_MACD[x]

        df_bollinger_cross_high = pd.DataFrame()
        df_bollinger_cross_low = pd.DataFrame()
        for x in data.columns:
            bollinger = ta.volatility.BollingerBands(data[x], window=20, window_dev=1.8)
            df_bollinger_cross_high[x] = bollinger.bollinger_hband_indicator()
            df_bollinger_cross_low[x] = bollinger.bollinger_lband_indicator()
            df_bollinger_cross_high[x] = pd.to_numeric(df_bollinger_cross_high[x], downcast='integer')
            df_bollinger_cross_low[x] = pd.to_numeric(df_bollinger_cross_low[x], downcast='integer')

        output = pd.DataFrame(df_RSI.stack(), columns=["rsi"]).merge(pd.DataFrame(df_MACD.stack(), columns=["macd"]),
                                                                      left_index=True, right_index=True).merge(
        pd.DataFrame(df_ema9_macd.stack(), columns=["df_ema9_macd"]), left_index=True, right_index=True).merge(
        pd.DataFrame(df_ema9_macd_hist.stack(), columns=["df_ema9_macd_hist"]), left_index=True,
        right_index=True).merge(pd.DataFrame(df_bollinger_cross_high.stack(), columns=["Bollinger_cross_high"]),
                                left_index=True, right_index=True).merge(
        pd.DataFrame(df_bollinger_cross_low.stack(), columns=["Bollinger_cross_low"]), left_index=True,
        right_index=True).reset_index().rename(columns={"level_1": "asset"})
        return output


    def ingest_tech(self, data):
        df_actual = pd.read_sql("SELECT * FROM cryptogroup2.tech_indikators ORDER BY date DESC LIMIT 1", self.engine_ds.connect())
        last_date = df_actual["date"][0]
        new_data = data[data["date"] > last_date]
        new_data.to_sql("tech_indikators", con=self.engine_ds, if_exists='append', index=False, chunksize=1000)




