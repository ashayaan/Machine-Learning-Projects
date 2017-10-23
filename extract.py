from bs4 import BeautifulSoup
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
	source_code = open('extract.html', 'r').read()
	soup = BeautifulSoup(source_code, "html.parser")
	# Extraacting data set names
	name = []
	for a in soup.find_all('a', {'class' : 'dataset-list-item__name'}):
		name.append(a.contents[2])
	
	# Extraacting data set link
	links = []

	for link in soup.find_all('a', {'class' : 'dataset-list-item__name'}, href=True):
		links.append(link['href'])

	x = {}

	# Extracting the dataset summary
	summary = []


	for s in soup.find_all('p', {'class' : 'dataset-list-item__summary'}):
		summary.append(s.text)

	tags = []
	for t in soup.find_all('p', {'class' : 'dataset-list-item__details'}):
		temp = []
		for y in t.find_all('a', {'class' : 'dataset-list-item__details--categories-link'}):
			temp.append(y.text)
		tags.append(temp)

	# tags = []
	# for t in soup.find_all('span',{'class' : 'dataset-list-item__details--categories'}):
	# 	s = t.text
	# 	l = s.split(',')
	# 	tags.append(l)

	for i in range(len(name)):
		print name[i],
		print ',',
		print links[i],
		print ',',
		for l in tags[i]:
			print l,
			print ',',
		print " "