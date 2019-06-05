import itertools
import pandas as pd
import json

TRANSACTIONS = [
    ["a", "b", "c", "d"],
    ["a", "b", "d", "e"],
    ["b", "d", "e"]
]

TRANSACTIONS2 = [
    ["a", "b", "c", "d"],
    ["b", "d", "f", "c"],
    ["e", "c"],
    ["a", "e"],
    ["b", "e"]
]


class PAR(object):

    def __init__(self, transactions):
        self.transactions = []
        self.occurrence_map = dict()
        self.cooccurrence_map = dict()
        for t in transactions:
            self.add_transaction(t)

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
            for p in agg:
                probs += [p[1] / p[2]]
                occs += [p[2]]
            agg_prob += [(k, sum(probs) * sum(occs))]
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
trained = PAR(df)
recommendation = trained.recommend(['TWTR', 'BBCT', 'PSFB', 'WFLG'])
print(recommendation)
trained.generate_vsjson(recommendation, ['TWTR', 'BBCT', 'PSFB', 'WFLG'])
# print(trained.recommend(['TAFF', 'PCSP', 'COLA', 'CRNT', 'BGMA', 'BEGG']))
