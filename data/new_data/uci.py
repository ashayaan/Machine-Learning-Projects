from bs4 import BeautifulSoup, Comment
import csv
import sys
import pandas as pd
import time
import dryscrape
import requests
reload(sys)
sys.setdefaultencoding('utf-8')


def extractData(batch):
	count = 0
	data = {} 
	for row in batch.find_all('tr'):
		count+=1
		if(count%2 != 0):
			continue
		flag = 0
		link = "https://archive.ics.uci.edu/ml/"+row.find("a",href=True).get( 'href')
		data_list = []
		for element in row.find_all('td'):
			if(flag == 0):
				flag+=1
				continue
			data_list.append(element.text)
		print data_list
		

if __name__ == '__main__':
	url = "https://archive.ics.uci.edu/ml/datasets.html"
	session = dryscrape.Session()
	try:
		print "Visiting " + url
		session.visit(url)
		response = session.body()
		# print session.body()
		soup = BeautifulSoup(response,"html.parser")
	except Exception as e:
		raise e
	
	soup.prettify()
	batch =  soup.find_all('table')[5]

	extractData(batch)