from collections import Counter

lis = [(3,100), (100, 4)]

a = max(lis, key=lambda x:x[1])

print(a)