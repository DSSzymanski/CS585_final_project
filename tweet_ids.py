import glob
import os

f = open('tweet_ids.txt', 'w')
path = 'Subtask_A'
id_list = dict()
total = 0
for filename in glob.glob(os.path.join(path, '*.txt')):
    per_file = 0
    with open(filename) as txt:
        lines = txt.read().split('\n')
        for line in lines:
            if len(line) > 0:
                l = line.split('\t')
                if l[0] not in id_list:
                    f.write(line + '\n')
                id_list[l[0]] = l[1]
                total += 1
                per_file += 1
    print(filename, per_file)
    txt.close()
f.close()
print(len(id_list), total, total - len(id_list))
            
