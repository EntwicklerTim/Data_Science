import pandas as pd
from sqlalchemy import create_engine
import ta

class TechInsert:
    def __init__(self,):
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')


    def calculate_tech_ind(self):
        #load prices
        data = pd.read_sql("SELECT * FROM cryptogroup2.HistCoins", self.engine_ds.connect())
        all_id = list(data["id"].unique())
        filt_price_id = {}
        for x in all_id:
            filt_price_id[x] = data[data["id"] == x].set_index("date")

        #initialize various DataFrames
        df_RSI = pd.DataFrame()
        df_MACD = pd.DataFrame()
        df_MACD_diff = pd.DataFrame()
        df_MACD_signal = pd.DataFrame()
        df_KAMA = pd.DataFrame()
        df_PPO = pd.DataFrame()
        df_PPO_hist = pd.DataFrame()
        df_PPO_signal = pd.DataFrame()
        df_STOCHrsi = pd.DataFrame()
        df_TSI = pd.DataFrame()
        df_FI = pd.DataFrame()
        df_OBV = pd.DataFrame()
        df_bollinger_cross_high = pd.DataFrame()
        df_bollinger_cross_low = pd.DataFrame()
        df_ULI = pd.DataFrame()
        df_Aroon = pd.DataFrame()
        df_kst = pd.DataFrame()
        df_STC = pd.DataFrame()
        df_PVO = pd.DataFrame()

        #calculate various technical indicators
        for x in all_id:
            rsi_21 = ta.momentum.RSIIndicator(filt_price_id[x]["price"], window=21)
            df_RSI[x] = rsi_21.rsi().dropna()
            macd_16 = ta.trend.MACD(filt_price_id[x]["price"], window_slow=26, window_fast=12, window_sign=9)
            df_MACD[x] = macd_16.macd()
            df_MACD_diff[x] = macd_16.macd_diff()
            df_MACD_signal[x] = macd_16.macd_signal()
            KAMA = ta.momentum.KAMAIndicator(filt_price_id[x]["price"], window=10, pow1=2, pow2=30, fillna=False)
            df_KAMA[x] = KAMA.kama().pct_change(20)
            PPO = ta.momentum.PercentagePriceOscillator(filt_price_id[x]["price"], window_slow=26, window_fast=12,
                                                        window_sign=9, fillna=False)
            df_PPO[x] = PPO.ppo().dropna()
            df_PPO_hist[x] = PPO.ppo_hist().dropna()
            df_PPO_signal[x] = PPO.ppo_signal().dropna()
            Stoch_rsi = ta.momentum.StochRSIIndicator(filt_price_id[x]["price"], window=14, smooth1=3, smooth2=3,
                                                      fillna=False)
            df_STOCHrsi[x] = Stoch_rsi.stochrsi().dropna()
            TSI = ta.momentum.TSIIndicator(filt_price_id[x]["price"], window_slow=25, window_fast=13, fillna=False)
            df_TSI[x] = TSI.tsi().dropna()
            FI = ta.volume.ForceIndexIndicator(close=filt_price_id[x]["price"], volume=filt_price_id[x]["volumes"],
                                               window=13, fillna=False)
            df_FI[x] = FI.force_index().dropna()
            OBV = ta.volume.OnBalanceVolumeIndicator(close=filt_price_id[x]["price"],
                                                     volume=filt_price_id[x]["volumes"], fillna=False)
            df_OBV[x] = OBV.on_balance_volume().pct_change(20).dropna()
            bollinger = ta.volatility.BollingerBands(filt_price_id[x]["price"], window=20, window_dev=1.8)
            df_bollinger_cross_high[x] = bollinger.bollinger_hband_indicator()
            df_bollinger_cross_low[x] = bollinger.bollinger_lband_indicator()
            df_bollinger_cross_high[x] = pd.to_numeric(df_bollinger_cross_high[x], downcast='integer')
            df_bollinger_cross_low[x] = pd.to_numeric(df_bollinger_cross_low[x], downcast='integer')
            ULI = ta.volatility.UlcerIndex(close=filt_price_id[x]["price"], window=14, fillna=False)
            df_ULI[x] = ULI.ulcer_index().dropna()
            aroon = ta.trend.AroonIndicator(close=filt_price_id[x]["price"], window=25, fillna=False)
            df_Aroon[x] = aroon.aroon_indicator().dropna()
            KST = ta.trend.KSTIndicator(close=filt_price_id[x]["price"], roc1=10, roc2=15, roc3=20, roc4=30, window1=10,
                                        window2=10, window3=10, window4=15, nsig=9, fillna=False)
            df_kst[x] = KST.kst().dropna()
            STC = ta.trend.STCIndicator(close=filt_price_id[x]["price"], window_slow=50, window_fast=23, cycle=10,
                                        smooth1=3, smooth2=3, fillna=False)
            df_STC[x] = STC.stc()
            PVO = ta.momentum.PercentageVolumeOscillator(filt_price_id[x]["volumes"], window_slow=26, window_fast=12,
                                                         window_sign=9, fillna=False)
            df_PVO[x] = PVO.pvo().dropna()


        #merge the various dataframes together
        output = pd.DataFrame(df_RSI.stack(), columns=["rsi"]).merge(
            pd.DataFrame(df_MACD.stack(), columns=["df_MACD"]), left_index=True, right_index=True).merge(
            pd.DataFrame(df_MACD_diff.stack(), columns=["df_MACD_diff"]), left_index=True, right_index=True).merge(
            pd.DataFrame(df_MACD_signal.stack(), columns=["df_MACD_signal"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_KAMA.stack(), columns=["df_KAMA"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_PPO.stack(), columns=["df_PPO"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_PPO_hist.stack(), columns=["df_PPO_hist"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_PPO_signal.stack(), columns=["df_PPO_signal"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_STOCHrsi.stack(), columns=["df_STOCHrsi"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_TSI.stack(), columns=["df_TSI"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_FI.stack(), columns=["df_FI"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_OBV.stack(), columns=["df_OBV"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_bollinger_cross_high.stack(), columns=["df_bollinger_cross_high"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_bollinger_cross_low.stack(), columns=["df_bollinger_cross_low"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_ULI.stack(), columns=["df_ULI"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_Aroon.stack(), columns=["df_Aroon"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_kst.stack(), columns=["df_kst"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_STC.stack(), columns=["df_STC"]), left_index=True,right_index=True).merge(
            pd.DataFrame(df_PVO.stack(), columns=["df_PVO"]), left_index=True,
            right_index=True).reset_index().rename(columns={"level_1": "asset"})

        return output


    def ingest_tech(self, data):
        #Load current data into the database
        df_actual = pd.read_sql("SELECT * FROM cryptogroup2.tech_indikator ORDER BY date DESC LIMIT 1", self.engine_ds.connect())
        last_date = df_actual["date"][0]
        new_data = data[data["date"] > last_date]
        new_data.to_sql("tech_indikator", con=self.engine_ds, if_exists='append', index=False, chunksize=1000)
