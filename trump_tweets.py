import re
import json

p = re.compile('http[s]?[\S]*')
emoji_pattern = re.compile('[\U00010000-\U0010ffff]*', flags=re.UNICODE)

total = 0
total_chars = 0
total_words = 0
f = open('trump_tweets.txt', 'w')
with open("trumptwitterarchive.json") as data:
    tweets = json.load(data)
    for tweet in tweets:
        line = tweet["text"].lower().replace('@', '').replace('#', '').replace('\n', ' ')
        line = p.sub('', line)
        line = emoji_pattern.sub('', line)
        f.write(line + '\n')
        total += 1
        total_chars += len(line)
        total_words += len(line.split())
f.close()
data.close()
ave_chars = total_chars/total
print('{} total tweets'.format(total))
print('{} total words, {} total characters, {} average characters per tweet'.format(total_words, total_chars, ave_chars))
