from collections import OrderedDict

with open('datagov_urls') as fin:
    lines = (line.rstrip() for line in fin)
    unique_lines = OrderedDict.fromkeys( (line for line in lines if line) )

url_list =  unique_lines.keys()
str_urls = reduce(lambda x,y:x+'\n'+y, url_list)
print str_urls
