import datetime
from datetime import timedelta
import pandas as pd
from coin_metrics_ingest import MetricInsert
from tech_ingest import TechInsert
from refutation.sentiment_corr_reject import Refute
from prediction_views.prediction import Predicition
gestern = pd.Timestamp(datetime.datetime.now().date() - timedelta(days=1))
import time
if __name__ == '__main__':
    # objekts
    metric_insert = MetricInsert()
    tech_insert = TechInsert()
    refute = Refute()
    predict = Predicition()
    predict.predict_tech()
    # ingest the tech data
    tech_insert.ingest_tech(tech_insert.calculate_tech_ind(tech_insert.load()))

    # ingest the metric data

    metric_insert.ingest_metric(metric_insert.load("RevHashNtv", ['btc', 'eth']))
    metric_insert.ingest_metric(metric_insert.load("AdrActCnt", ['btc', 'eth', 'ltc']))
    metric_insert.ingest_metric(metric_insert.load("SplyFF", ['btc', 'ada']))
    metric_insert.ingest_metric(metric_insert.load("DiffLast", ['eth']))
    metric_insert.ingest_metric(metric_insert.load("BlkSizeMeanByte", ['eth', 'ltc', 'ada']))
    metric_insert.ingest_metric(metric_insert.load("NDF", ['eth', 'ltc']))
    metric_insert.ingest_metric(metric_insert.load("SER", ['ada', 'eth']))
    metric_insert.ingest_metric(metric_insert.load("TxCnt", ['eth', 'ltc', 'ada', 'xmr']))
    metric_insert.ingest_metric(metric_insert.load("IssTotNtv", ['xmr']))
    metric_insert.ingest_metric(metric_insert.load("RevNtv", ['xmr']))
    time.sleep(60)
    #make predictions
    predict.predict_tech()
    #predict.predict_metric_price()
