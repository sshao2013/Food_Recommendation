from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql import SparkSession
import pandas as pd

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]


class AR(object):

    def __init__(self, transactions):
        self.transactions = transactions
        self.total = len(self.transactions)
        self.rules = self.arRules(self.transactions)

    def arRules(self, transaction):
        spark = SparkSession.builder.getOrCreate()

        R = Row('ID', 'items')  # use enumerate to add the ID column
        df = spark.createDataFrame([R(i, x) for i, x in enumerate(transaction)])
        fpGrowth = FPGrowth(itemsCol='items', minSupport=0.0001, minConfidence=0.0001)
        model = fpGrowth.fit(df)
        rules = model.associationRules.collect()  # Display generated association rules.
        return rules

    def recommend(self, transactions):
        transactions = list(set(transactions))
        rec_dict = dict()
        for singleRule in self.rules:
            if any(elem in singleRule[1] for elem in transactions):
                continue;
            if any(elem in singleRule[0] for elem in transactions):
                count_interaction = sum(el in singleRule[0] for el in transactions)
                match_score = ((count_interaction ** 2) / (len(singleRule[0]) * len(transactions))) * singleRule[2]
                for item in singleRule[1]:
                    if item not in rec_dict:
                        rec_dict[item] = match_score
                    else:
                        rec_dict[item] += 1

        return list(rec_dict.items())


trained = AR(TRANSACTIONS)
print(trained.recommend('a'))

