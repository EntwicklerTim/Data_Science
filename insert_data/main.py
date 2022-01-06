from coin_metrics_ingest import MetricInsert
from tech_ingest import TechInsert
from prediction_views.prediction import Predicition

if __name__ == '__main__':
    #objekts
    metric_insert = MetricInsert()
    tech_insert = TechInsert()

    # ingest the tech data
    data = tech_insert.calculate_tech_ind()
    tech_insert.ingest_tech(data)
    #predict tech data
    predict = Predicition()
    predict.predict_tech()

    # ingest the metric data
    metric_insert.load()
