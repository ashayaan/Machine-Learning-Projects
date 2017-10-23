from bs4 import BeautifulSoup
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
	source_code = open('extract.html', 'r').read()
	soup = BeautifulSoup(source_code, "html.parser")
	# Extracting data set names
	name = []
	for a in soup.find_all('a', {'class' : 'dataset-list-item__name'}):
		s = a.contents[2]
		s = s.replace(',', '')
		name.append(s)
	
	# Extracting data author
	author=[]
	for a in soup.find_all('a', {'class' : 'dataset-list-item__details--author'}):
		author.append(a.text)


	# Extracting data set link
	links = []
	for link in soup.find_all('a', {'class' : 'dataset-list-item__name'}, href=True):
		links.append(link['href'])

	x = {}

	# Extracting the dataset summary
	summary = []
	for s in soup.find_all('p', {'class' : 'dataset-list-item__summary'}):
		x = s.text
		x = x.replace(',', '')
		summary.append(x)

	# Extracting the number of downloads
	downloads = []
	for s in soup.find_all('span', {'class' : 'dataset-list-item__metadatum--nonlink'}):
		l = s.text.split()
		l[0] = l[0].replace(',', '')
		downloads.append(l[0])

	#Extracting time
	posttime = []
	for t in soup.find_all('p', {'class' : 'dataset-list-item__details'}):
		for y in t.find_all('span'):
			val = y.get('title')
			if val is not None:
				posttime.append(val)

	#Extracting tags
	tags = []
	for t in soup.find_all('p', {'class' : 'dataset-list-item__details'}):
		temp = []
		for y in t.find_all('a', {'class' : 'dataset-list-item__details--categories-link'}):
			temp.append(y.text)
		tags.append(temp)


	print "Name,Author,Link,Time,Summary,Downloads,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6"
	for i in range(len(name)):
		print name[i],
		print ',',
		print author[i],
		print ',',
		print 'www.kaggle.com'+links[i],
		print ',',
		print str(posttime[i]),
		print ',',
		print str(summary[i]),
		print ',',
		print downloads[i],
		print ',',
		for l in tags[i]:
			print l,
			print ',',
		print " "