from shutil import copy2
from os import listdir
from os.path import isfile, join

src = '/Users/savyasachi/Downloads/UTKFace/'
dst = '/Users/savyasachi/utk/'


def getEthnicity(a):
    return a.split('_')[2]


def copyfile(file):
    eth = getEthnicity(file)
    copy2(src + file, dst + eth + '/')

onlyfiles = [f for f in listdir(src) if isfile(join(src, f))]
wrong_files = []

count = [0,0,0,0,0]

for file in onlyfiles:
    try:
        eth = int(getEthnicity(file))
        count[eth] += 1
    except ValueError:
        wrong_files.append(file)

print(count)