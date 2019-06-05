import pandas as pd

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import Row
from pyspark.sql import SparkSession

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

TRANSACTIONS2 = [
    ["a", "b", "c", "d"],
    ["b", "a", "d", "f"],
    ["d", "e"]
]

class PAR_New(object):

    def __init__(self, transactions):
        self.transactions = transactions
        self.total_count = len(self.transactions)
        self.par_result = self.PAR(self.transactions)

    def add_transaction(self, transaction):
        transaction = list(set(transaction))
        self.transactions.append(transaction)
        self.total_count = len(self.transactions)
        self.par_result = self.PAR(self.transactions)

    def PAR(self, transaction):
        MAX_MEMORY = "8g"
        spark = SparkSession.builder.config("spark.executor.memory", MAX_MEMORY).config("spark.driver.memory",
                                                                                        MAX_MEMORY).getOrCreate()

        R = Row('ID', 'items') # use enumerate to add the ID column
        df = spark.createDataFrame([R(i, x) for i, x in enumerate(transaction)])

        fp_growth = FPGrowth(itemsCol='items', minSupport=0.00001, minConfidence=0.00001)
        freq = fp_growth.fit(df).freqItemsets.collect()
        supp_x = sorted(list(filter(lambda x: len(x[0]) == 1, freq)))
        supp_xy = sorted(list(filter(lambda x: len(x[0]) == 2, freq)))

        # Rule Power Factor (RPF)
        par_result = dict()
        for i, j in supp_x:
            if(i[0]!= '$MISS'):
                par_result[i[0]] = dict()
                for m, n in supp_xy:
                    if m[0] == i[0] and m[1]!= '$MISS':
                        par_result[i[0]][m[1]] = ((n / self.total_count) ** 2) / (j / self.total_count)
                    elif m[1] == i[0] and m[0]!= '$MISS':
                        par_result[i[0]][m[0]] = ((n / self.total_count) ** 2) / (j / self.total_count)

        return {k: v for k, v in par_result.items() if len(v) > 0}

    def recommend(self, transactions):
        final_result = dict()
        transactions = list(set(transactions))
        for item in transactions:
            if self.par_result.get(item) is not None:
                for k, v in self.par_result.get(item).items():
                    if k not in transactions:
                        if final_result.get(k) is None:
                            final_result.update({k: v})
                        else:
                            final_result[k] += v

        return sorted([(k, v) for k, v in final_result.items()], key=lambda x: x[0], reverse=False)


# df = pd.read_pickle('meal_data.pkl')
# df = df['food_codes'].tolist()
# trained = PAR_New(df)
# print(trained.recommend(['TAFF', 'PCSP', 'COLA', 'CRNT', 'BGMA', 'BEGG']))

trained = PAR_New(TRANSACTIONS2)
# print(trained.recommend('a'))
# print(trained.recommend('b'))
# print(trained.recommend('c'))
print(trained.recommend(['a', 'b', 'd']))
# print(trained.recommend(['a', 'b', 'c', 'd']))
# print(trained.recommend(['a', 'f']))
