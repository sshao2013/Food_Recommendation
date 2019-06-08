import itertools
import pandas as pd

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

TRANSACTIONS2 = [
    ['a', 'b'], ['a', 'd', 'c']
]


class PAR(object):

    def __init__(self, transactions):
        self.transactions = []
        self.occurrence_map = dict()
        self.cooccurrence_map = dict()
        for t in transactions:
            self.add_transaction(t)
        self.total_count = len(transactions)

    def add_transaction(self, transaction):
        transaction = list(set(transaction))
        self.transactions += transaction

        for t in transaction:
            oc_f1 = self.occurrence_map.get(t, 0)
            oc_f1_d = dict()
            oc_f1_d[t] = oc_f1 + 1
            self.occurrence_map.update(oc_f1_d)

        for f1, f2 in itertools.product(transaction, repeat=2):
            if f1 != f2:
                cooc_f1 = self.cooccurrence_map.get(f1, dict())
                cooc_f1_f2 = cooc_f1.get(f2, 0)
                cooc_f1_f2_d = dict()
                cooc_f1_f2_d[f2] = cooc_f1_f2 + 1
                cooc_f1.update(cooc_f1_f2_d)
                cooc_d = dict()
                cooc_d[f1] = cooc_f1
                self.cooccurrence_map.update(cooc_d)

    def recommend(self, transaction):
        transaction = list(set(transaction))
        oc_map = self.occurrence_map
        cc_map = self.cooccurrence_map
        recs = [
            (k, v, oc_map.get(t), oc_map.get(k))
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
            result_p = []
            convic = []
            for p in agg:
                probs += [max(p[1] / p[2], p[1]/p[3])]
                occs += [p[2]/self.total_count]
                result_p += [p[1]]
                rpf += [p[1] * p[1] / p[2]]
                convic += [(1 - p[3] / self.total_count) / (1 - p[1] / p[2])]
            agg_prob += [(k, sum(probs) * sum(occs) + sum(rpf))]
        agg_prob = sorted(agg_prob, key=lambda x: x[1], reverse=True)
        return agg_prob


# df = pd.read_pickle('meal_data.pkl')
# df = df['food_codes'].tolist()
# trained = PAR(df)
# print(trained.recommend(['WALK', 'TNGS', 'OASI', 'WGPS']))

trained = PAR(TRANSACTIONS2)
print(trained.recommend(['b', 'd']))

