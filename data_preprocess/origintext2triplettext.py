from collections import defaultdict
import pickle
"""
process one line in file "origintext.txt"
"""
BASE_DIR = "../data/"
def parse_one_line(line):
    line = line.strip().split('\t')
    pv_triples = defaultdict(list)
    pv_pairs1 = line[5]
    pv_pairs2 = line[9]
    label = line[10]

    # process pv_pairs1
    for pv_pair in pv_pairs1.split(';'):
        if len(pv_pair.split(":")) != 2:
            continue
        p = pv_pair.split(':')[0].strip("#")
        v = pv_pair.split(':')[1].strip("#")
        """
        in case that for a specific property, 
        there exists multiple values in this item for this property
        the process logit here is  that we concatenate all these values
        and the final concatenated values stands for a new value for this property in this item
        here we use the delimiter "#" as the concatenate symbol
        """
        if p in pv_triples.keys():
            pv_triples[p][0] = pv_triples[p][0] + "#" + v
        else:
            pv_triples[p].append(v)

    # process pv_pairs2
    for pv_pair in pv_pairs2.split(";"):
        if len(pv_pair.split(":")) != 2:
            continue
        p = pv_pair.split(':')[0].strip("#")
        v = pv_pair.split(":")[1].strip("#")

        """
        in case that item2 has a property that item1 do not have, 
        we sholud skip this property
        else if item2 has already push a value in this property, we should concatenate this new value
        else we push this value directly
        """
        if p not in pv_triples.keys():
            continue
        elif len(pv_triples[p]) == 2:
            pv_triples[p][1] = pv_triples[p][1] + "#" + v
        else:
            pv_triples[p].append(v)
    sample = []
    for p, v in pv_triples.items():
        # in case that for a specific property,
        # only item1 exist a value for this property, we should skip directly
        if len(v) == 2:
            sample.append([p] + v)
    sample.append(label)
    return sample
def ProcessOriginText(filename=BASE_DIR + "./origintext.txt"):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            sample = parse_one_line(line)
            data.append(sample)

    pickle.dump(data, open(BASE_DIR + "triplettext.pkl", "wb"))

    return data


if __name__ == "__main__":
    ProcessOriginText()