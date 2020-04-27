import pickle
from copy import deepcopy
BASE_DIR = "../data/"
property2id = {}
id2property = {}
value2id = {}
id2value = {}
entity2id = {}
id2entity = {}
tripletid = []
tupleid = []
"""
        each instance in "triplettext.txt" like follow
        [['系列', '千张包', '千张包'], 
        ['口味', '传统（真空抽气包装）', '原味'], 
        ['品牌', '丁莲芳', '丁莲芳'], 
        ['产地', '中国大陆', '中国大陆'], 
        ['包装方式', '包装', '包装'], 
        ['重量', '260', '7800'], 
        ['省份', '浙江省', '浙江省'],
        "0"]
"""
triplettext = pickle.load(open(BASE_DIR + "triplettext.pkl", "rb"))
for instance in triplettext:
    for triplet in instance:

        """
        if we encounter the label, then we should end this instance
        """
        if len(triplet) == 1:
            break
        P, V1, V2 = triplet
        if P not in property2id:
            property2id[P] = len(property2id)
            id2property[len(id2property)] = P
total_propertys = len(property2id)
entity2id = deepcopy(property2id)
id2entity = deepcopy(id2property)
for instance in triplettext:
    for triplet in instance:
        if len(triplet) == 1:
            break
        P, V1, V2 = triplet
        """
        we can endow the base ids for each value
        """
        if V1 not in value2id.keys() and V1 not in property2id.keys():
            value2id[V1] = len(value2id) + total_propertys
        if V2 not in value2id.keys() and V2 not in property2id.keys():
            value2id[V2] = len(value2id) + total_propertys


"""
since value ids are based on property ids
so we should reorder the ids of values based on the base ids of values
and at the same time we can have a full entity2id and id2entity dict
"""
for value, id in value2id.items():
    if value in property2id.keys():
        print(value)
    entity2id[value] = id
    id2entity[id] = value

"""
now we can transfer triplettext to tripletid
"""
for instance in triplettext:

    Triid = []
    Tupid = []
    for triplet in instance:
        """
        if we encounter the label, then we should end this instance
        """
        if len(triplet) == 1:
            Triid.append(int(triplet[0]))
            Tupid.append(int(triplet[0]))
            break
        PID, VID1, VID2 = entity2id[triplet[0]], entity2id[triplet[1]], entity2id[triplet[2]]
        Triid.append([PID, VID1, VID2])
        """
        if two items have different value under this property, 
        then we set value as 0
        else we set value as corresponding value id
        """
        if VID1 == VID2:
            Tupid.append([PID, VID1])
        else:
            Tupid.append([PID, 0])
    tripletid.append(Triid)
    tupleid.append(Tupid)

pickle.dump(property2id, open(BASE_DIR + "property2id", "wb"))
pickle.dump(value2id, open(BASE_DIR + "value2id", "wb"))
pickle.dump(entity2id, open(BASE_DIR + "entity2id", "wb"))
pickle.dump(id2property, open(BASE_DIR + "id2property", "wb"))
pickle.dump(id2value, open(BASE_DIR + "id2value", "wb"))
pickle.dump(id2entity, open(BASE_DIR + "id2entity", "wb"))
pickle.dump(tripletid, open(BASE_DIR + "tripletid", "wb"))
pickle.dump(tupleid, open(BASE_DIR + "tupleid", "wb"))
