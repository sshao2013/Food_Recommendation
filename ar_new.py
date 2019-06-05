from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql import SparkSession
import pandas as pd

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

MAX_MEMORY = "8g"


class AR(object):

    def __init__(self, transactions):
        self.result = self.arRules(transactions)

    def arRules(self, transaction):
        spark = SparkSession.builder.config("spark.executor.memory", MAX_MEMORY).config("spark.driver.memory", MAX_MEMORY).getOrCreate()

        R = Row('ID', 'items')  # use enumerate to add the ID column
        df = spark.createDataFrame([R(i, x) for i, x in enumerate(transaction)])
        fpGrowth = FPGrowth(itemsCol='items', minSupport=0.001, minConfidence=0.001)
        model = fpGrowth.fit(df)
        return model

    def recommend(self, transactions):
        spark = SparkSession.builder.config("spark.executor.memory", MAX_MEMORY).config("spark.driver.memory", MAX_MEMORY).getOrCreate()
        R = Row('items')
        df = spark.createDataFrame([R(transactions)])
        result = self.result.transform(df).collect()
        # format the result: final_result
        final_result = []
        for item in result:
            count = len(item[1])
            for rec in item[1]:
                final_result.append((rec, count))
                count -= 1

        return final_result


# df = pd.read_pickle('meal_data.pkl')
# df = df['food_codes'].tolist()
# trained = AR(df)
# print(trained.recommend(['TAFF', 'PCSP', 'COLA', 'CRNT', 'BGMA', 'BEGG']))


trained = AR(TRANSACTIONS)
print(trained.recommend(['b','d']))
# print(trained.recommend(["a", "b", "d"]))
# print(trained.recommend('b'))
# print(trained.recommend('c'))
# print(trained.recommend('d'))
#
# print(trained.recommend(['a', 'd', 'b', 'c']))

# new_trans = ["a", "b", "d"]
# trained.add_transaction(new_trans)
# print(trained.recommend('a'))
