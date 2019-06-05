import itertools
import json
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
    ["b", "a", "d", "c"],
    ["d", "e"]
]


class PAR_New(object):

    def __init__(self, transactions):
        self.transactions = transactions
        self.total_count = len(self.transactions)
        self.occurrence_map, self.cooccurrence_map = self.PAR(self.transactions)

    def add_transaction(self, transaction):
        transaction = list(set(transaction))
        self.transactions.append(transaction)
        self.total_count = len(self.transactions)
        self.par_result = self.PAR(self.transactions)

    def PAR(self, transaction):
        MAX_MEMORY = "10g"
        spark = SparkSession.builder.master("local").config("spark.memory.fraction", 0.8) \
            .config("spark.executor.heartbeatInterval", "1000") \
            .config("spark.network.timeout", "1200") \
            .config("spark.executor.memory", "14g") \
            .config("spark.driver.memory", "14g") \
            .config("spark.executor.extraJavaOptions", "Xmx8024m") \
            .config("spark.sql.shuffle.partitions", "1000") \
            .config("spark.driver.maxResultSize", "0") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.executor.cores", "4").getOrCreate()

        R = Row('ID', 'items')  # use enumerate to add the ID column
        df = spark.createDataFrame([R(i, x) for i, x in enumerate(transaction)])

        fp_growth = FPGrowth(itemsCol='items', minSupport=(0.001), minConfidence=(0.001), numPartitions=200)
        freq = fp_growth.fit(df).freqItemsets.collect()
        rule = fp_growth.fit(df).associationRules.collect()

        supp_x = sorted(list(filter(lambda x: len(x[0]) == 1, freq)))
        supp_xy = sorted(list(filter(lambda x: len(x[0]) == 2, freq)))
        supp_x = {k[0]: v for k, v in supp_x if k[0] != '$MISS'}
        supp_xy = list(filter(lambda k: k[0][0] != '$MISS' and k[0][1] != '$MISS', supp_xy))

        par_result = dict()

        for i, j in supp_x.items():
            if (i != '$MISS'):
                par_result[i] = dict()
                for m, n in supp_xy:
                    if m[0] == i and m[1] != '$MISS':
                        par_result[i][m[1]] = (((n / len(transaction)) ** 2) / (j / len(transaction)), n)
                    elif m[1] == i and m[0] != '$MISS':
                        par_result[i][m[0]] = (((n / len(transaction)) ** 2) / (j / len(transaction)), n)

        return supp_x, {k: v for k, v in par_result.items() if len(v) > 0}

    def recommend(self, transactions):
        transaction = list(set(transactions))
        oc_map = self.occurrence_map
        cc_map = self.cooccurrence_map
        recs = [
            (k, v, oc_map.get(t))
            for t in transaction
            for k, v in cc_map.get(t, dict()).items()
            if k not in transaction
        ]
        recs = [r for r in recs if r[2] is not None]
        # Grouped is a generator, so a single comprehension is not working
        agg_prob = []
        for k, agg in itertools.groupby(sorted(recs, key=lambda x: x[0]), lambda x: x[0]):
            probs = []
            occs = []
            rpf = []
            for p in agg:
                probs += [p[1][1] / p[2]]
                occs += [p[2]]
                rpf += [p[1][0]]
            agg_prob += [(k, sum(probs) * sum(occs) + sum(rpf))]

        agg_prob = sorted(agg_prob, key=lambda x: x[1], reverse=True)
        return agg_prob

    def generate_vsjson(self, recommend, transaction):

        data = {}
        data['nodes'] = []
        data['links'] = []
        data['nodes'].append({
            'id': ','.join(transaction),
            'group': 1,
            'hgt': 8})
        count = 0
        length_bia = 0
        for item in recommend:
            if count < 5:
                group_num = 2
                length_bia = 5
            elif count < 15:
                group_num = 3
                length_bia = 50
            else:
                group_num = 4
                length_bia = 100
            data['nodes'].append({
                'id': item[0],
                'group': group_num
            })
            data['links'].append({
                'source': ','.join(transaction),
                'target': item[0],
                'value': length_bia
            })
            count += 1
            if count > 30:
                break

        with open('/Users/sshao/Documents/NCL/CS8499/RS/demo/datasets/datatest.json', 'w') as outfile:
            json.dump(data, outfile)


df = pd.read_pickle('meal_data.pkl')
df = df['food_codes'].tolist()
trained = PAR_New(df)
recommendation = trained.recommend(['TWTR', 'BBCT', 'PSFB', 'WFLG'])
print(recommendation)
trained.generate_vsjson(recommendation, ['TWTR', 'BBCT', 'PSFB', 'WFLG'])
# print(trained.recommend(['TWTR', 'BBCT', 'PSFB', 'WFLG']))
# print(len(trained.recommend(['TWTR', 'BBCT', 'PSFB', 'WFLG'])))
# print(trained.recommend(['TAFF', 'PCSP', 'COLA', 'CRNT', 'BGMA', 'BEGG']))

# trained = PAR_New(TRANSACTIONS2)
# trained.generate_vsjson(trained.recommend(['a', 'b', 'd']), ['a', 'b', 'd'])
# print(trained.recommend('a'))
# print(trained.recommend('b'))
# print(trained.recommend('c'))
# print(trained.recommend(['a', 'b', 'd']))
# print(trained.recommend(['a', 'b', 'c', 'd']))
# print(trained.recommend(['a', 'f']))
