import os

convFolder = "archive_converted_32"
# initialize the class labels
names = os.listdir(convFolder)
print(names)
print(len(names))
names = list(filter(lambda x: not x.startswith('.'), names))
