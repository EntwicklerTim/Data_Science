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


    def load(self, metrics, asset_list):
        input_metric = self.client_coinMetric.get_asset_metrics(
            assets=asset_list,
            metrics=metrics,
            frequency=self.frequency,
            start_time=self.start_time,
            end_time=self.end_time
        ).to_dataframe()


        input_metric["date"] = pd.to_datetime(input_metric['time']).dt.date
        input_metric.sort_values(by = ["date"],inplace = True)
        input_metric["metric_name"] = metrics
        input_metric["metric"] = input_metric[metrics].astype('float64')#.round(5)
        input_metric.drop([metrics], axis=1, inplace=True)
        input_metric = input_metric[["date","asset","metric_name","metric"]]
        input_metric.replace(({"btc":"bitcoin", "eth": "ethereum","ltc": "litecoin","ada":"cardano", "xmr": "monero" }) ,inplace = True)

        return input_metric , metrics


    def ingest_metric(self, data_metric):
        metric = data_metric[1]
        df_actual = pd.read_sql("SELECT * FROM cryptogroup2.CoinMetric where metric_name = %(metric_para)s  ORDER BY date DESC LIMIT 1", self.engine_ds.connect(), params={"metric_para":metric})
        last_date = df_actual["date"][0]
        data = data_metric[0]
        new_data = data[data["date"] > last_date]
        new_data.to_sql("CoinMetric", con=self.engine_ds, if_exists='append', index=False, chunksize=1000)  # if i want all data than  if_exists='replac'