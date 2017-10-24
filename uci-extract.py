import urllib2
from bs4 import BeautifulSoup
import csv
import sys
import re
reload(sys)
sys.setdefaultencoding('utf-8')


def extractData(soup):
	data = {}
	name = []
	links = []
	
	for par in soup.find_all('p',{'class':'normal'}):
		for na in par.find_all('a',href=True):
			comma = na.text
			comma = comma.replace(',', '')
			name.append(comma)
			links.append(na['href'])
	
	for i in range(30,len(name)):
		data[name[i]] = [str('https://archive.ics.uci.edu/ml/' + links[i])]
	
	f = open('data2.csv','w')
	f.write("name,link,date,category\n")

	for key in data.keys():
		s = data[key][0]
		r = urllib2.urlopen(s)
		soup2 = BeautifulSoup(r,'lxml')
		sys.stdout.write("running for" +'\t'+str(key)+'\n')
		# print "runnig for " + 
		if(soup2.find('table',{'border':'1'})):
			data[key].append(soup2.find('table',{'border':'1'}).find_all('td')[11].text)
			data[key].append(soup2.find('table',{'border':'1'}).find_all('td')[5].text)
		else:
			data[key].append('N/A')
			data[key].append('N/A')
		f.write(key)
		f.write(',')
		f.write(data[key][0])
		f.write(',')
		f.write(data[key][1])
		f.write(',')
		f.write(data[key][2]+'\n')

		

if __name__ == '__main__':
	source_code = open('temp.html', 'r').read()
	soup = BeautifulSoup(source_code, "html.parser")
	extractData(soup)

