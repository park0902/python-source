from collections import Counter
from konlpy.tag import Hannanum
import pytagcloud

f = open('d://dd7.txt', 'r', encoding='UTF-8')
data = f.read()

nlp = Hannanum()
nouns = nlp.nouns(data)

count = Counter(nouns)
tags2 = count.most_common(200)
taglist = pytagcloud.make_tags(tags2, maxsize=80)
pytagcloud.create_tag_image(taglist, 'd://wordcloud.jpg', size=(900, 600), fontname='korean', rectangular=False)

f.close()