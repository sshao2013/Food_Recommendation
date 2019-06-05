import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql import SparkSession

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

# TRANSACTIONS2 = [
#     ['a','b','c'],['a','d','c']
# ]

TRANSACTIONS2 = [
    ["a", "b", "c"],
    ["b", "a", "d"],
    ["d", "e"],
]


class PAR_Third(object):

    def __init__(self, transactions):
        self.transactions = transactions
        self.total_count = len(self.transactions)
        self.rule_list, self.freq_list = self.PAR(self.transactions)

    def PAR(self, transaction):
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

        rules_filt = [
            (item[0], item[1], item[2])
            for item in self.rule_list
            if all(elem in transaction for elem in item[0])
        ]

        record = []
        for item in rules_filt:
            if (item[1][0] == 'PASS'):
                record.append(item)

        result_conf = dict()
        result_x = dict()
        for item in rules_filt:
            xy = 0
            for f in recs:
                if sorted(f[0]) == sorted(item[0]):
                    xy += f[1]
                    break
            conf = []
            for k in item[1]:
                if k not in transaction:
                    if k in result_conf:
                        result_conf[k] += item[2]
                        result_x[k] += xy
                        # result[k] += xy/self.total_count * item[2] + len(item[0])
                    else:
                        result_conf[k] = item[2]
                        result_x[k] = xy
                        # result[k] = xy/self.total_count * item[2] + len(item[0])
        result = dict()
        for item in result_conf.items():
            result[item[0]] = item[1] * result_x[item[0]]
        return sorted([(k, v) for k, v in result.items()], key=lambda x: x[1], reverse=True)


df = pd.read_pickle('meal_data.pkl')
df = df['food_codes'].tolist()
trained = PAR_Third(df)
print(trained.recommend(['BANA', 'TWTR', 'BTEA', 'SMLK']))
# print(trained.recommend(['WALK', 'TNGS', 'OASI', 'WGPS']))


# trained = PAR_Third(TRANSACTIONS2)
# print(trained.recommend('a'))
# print(trained.recommend('b'))
# print(trained.recommend('c'))
# print(trained.recommend(['a', 'b', 'd']))
# print(trained.recommend(['a', 'b', 'c', 'd']))
# print(trained.recommend(['a', 'd']))
