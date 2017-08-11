import re

file_post = open('data/small/weibo_pair_train_Q.post', 'rb')
file_resp = open('data/small/weibo_pair_train_Q.response', 'rb')

symbol = {}
for line in file_post.readlines():
    line = line.decode('utf-8')[:-1]
    for word in re.split(' ', line):
        if word in symbol:
            symbol[word] += 1
        else:
            symbol[word] = 1
for line in file_resp.readlines():
    line = line.decode('utf-8')[:-1]
    for word in re.split(' ', line):
        if word in symbol:
            symbol[word] += 1
        else:
            symbol[word] = 1

file_result = open('result.txt', 'wb')

total = sum([x[1] for x in symbol.items()])
num = 0
word_num = 0

file_result.write('<go>\n<eos>\n<unk>\n<pad>\n')

for item in sorted(symbol.items(), key=lambda x:x[1], reverse=True):
    file_result.write(item[0].encode('utf-8') + '\n')
    num += item[1]
    word_num += 1
    if num >= total * 0.99:
        break

print word_num
