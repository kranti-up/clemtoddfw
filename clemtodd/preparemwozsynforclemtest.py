import json


def readfile(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    return [json.loads(line) for line in data]



#data = readfile("synthetic_goals.jsonl")
data = readfile("synthetic_goals_unrealistic.jsonl")
dump_data = []
for d in data:
    dump_data.append(d)
#with open("subset_synthetic_taskdata_test.json", "w") as f:
with open("subset_synthetic_taskdata_test_unrealistic.json", "w") as f:
    json.dump(dump_data, f, indent=4)