from collections import defaultdict

dict = defaultdict()
dict.default_factory = int
dict[21] = 10
dict[22] = 11
dict[23] = 111
dict[24] = 1
dict[25] += 10
dict[22] += 100

print dict
print max(dict, key=dict.get)
