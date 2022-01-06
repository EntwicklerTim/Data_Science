import datetime
from coinmetrics.api_client import CoinMetricsClient
import pandas as pd
from sqlalchemy import create_engine


class MetricInsert:

    def __init__(self):
        self.engine_ds = create_engine('mysql+mysqlconnector://ds_mariadb:pw_ds_mariadb@85.214.56.217:3306/cryptogroup2')
        self.client_coinMetric = CoinMetricsClient()
        self.frequency = "1d"
        self.start_time = "2017-12-02"
        self.end_time = str(datetime.datetime.now().date())


    def load(self):
        #all choosen metrics
        metric_dict = {
            "RevHashNtv": ['eth'],
            "AdrActCnt": ['btc', 'eth', 'ltc'],
            "SplyFF": ['btc', 'ada'],
            "DiffLast": ['eth'],
            "BlkSizeMeanByte": ['eth', 'ltc', 'ada'],
            "NDF": ['eth', 'ltc'],
            "SER": ['ada', 'eth'],
             "TxCnt": ['eth', 'ltc', 'ada', 'xmr'],
             "IssTotNtv": ['xmr'],
             "RevNtv": ['xmr']
        }
        #load metric
        for x in metric_dict:
            input_metric = self.client_coinMetric.get_asset_metrics(
                assets=metric_dict[x],
                metrics=x,
                frequency=self.frequency,
                start_time=self.start_time,
                end_time=self.end_time
            ).to_dataframe()

            input_metric["date"] = pd.to_datetime(input_metric['time']).dt.date
            input_metric.sort_values(by = ["date"],inplace = True)
            input_metric["metric_name"] = x
            input_metric["metric"] = input_metric[x].astype('float64')#.round(5)
            input_metric.drop([x], axis=1, inplace=True)
            input_metric = input_metric[["date","asset","metric_name","metric"]]
            input_metric.replace(({"btc":"bitcoin", "eth": "ethereum","ltc": "litecoin","ada":"cardano", "xmr": "monero" }) ,inplace = True)

            #Load current data into the database
            df_actual = pd.read_sql("SELECT * FROM cryptogroup2.CoinMetric where metric_name = %(metric_para)s  ORDER BY date DESC LIMIT 1", self.engine_ds.connect(), params={"metric_para":x})
            last_date = df_actual["date"][0]
            data = input_metric
            new_data = data[data["date"] > last_date]
            new_data.to_sql("CoinMetric", con=self.engine_ds, if_exists='append', index=False, chunksize=1000)