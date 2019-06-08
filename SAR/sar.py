import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql import SparkSession

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

class SAR(object):

    def __init__(self, transactions):
        self.transactions = transactions
        self.total_count = len(self.transactions)
        self.rule_list, self.freq_list = self.SAR(self.transactions)

    def SAR(self, transaction):
        MAX_MEMORY = "12g"
        spark = SparkSession.builder.master("local").config("spark.memory.fraction", 0.8) \
            .config("spark.executor.memory", MAX_MEMORY) \
            .config("spark.driver.memory", MAX_MEMORY).getOrCreate()

        R = Row('ID', 'items')  # use enumerate to add the ID column
        df = spark.createDataFrame([R(i, x) for i, x in enumerate(transaction)])

        fp_growth = FPGrowth(itemsCol='items', minSupport=(0.001), minConfidence=(0.001), numPartitions=100)
        df_fit = fp_growth.fit(df)

        freq = df_fit.freqItemsets.collect()
        freq_list = list(filter(lambda x: len(x[0]) > 1, freq))

        rule = df_fit.associationRules.collect()
        rule_list = list(filter(lambda x: x[3] > 1, rule))
        return rule_list, freq_list

    def recommend(self, transactions):
        transaction = list(set(transactions))

        recs = [
            (k, v)
            for k, v in self.freq_list
            if any(elem in transaction for elem in k)
        ]

        result = dict()
        for item in self.rule_list:
            if all(elem in transaction for elem in item[0]):
                xy = 0
                for f in recs:
                    if sorted(f[0]) == sorted(item[0] + item[1]):
                        xy += f[1]
                        break
                for k in item[1]:
                    if k not in transaction:
                        if k in result:
                            #[k] += xy * item[2] * (len(item[0]) / len(transaction))
                            result[k] += item[2]
                        else:
                            #result[k] = xy * item[2] * (len(item[0]) / len(transaction))
                            result[k] = item[2]
        return sorted([(k, v) for k, v in result.items()], key=lambda x: x[1], reverse=True)


# df = pd.read_pickle('meal_data.pkl')
# df = df['food_codes'].tolist()
# trained = PAR_Third(df)
# print(trained.recommend(['WALK', 'TNGS', 'OASI', 'WGPS']))

trained = SAR(TRANSACTIONS)
print(trained.recommend(['a', 'b']))