from bs4 import BeautifulSoup
import csv
import sys
import requests
import pandas as pd
import time
import dryscrape
reload(sys)
sys.setdefaultencoding('utf-8')

data_list= []

def extractData(soup):
	x = soup.find_all("table",{"class":"wikitable sortable"})
	for table in x:
		for row in table.find_all("tr")[1:]:
			print row
			data={}
			temp = row.find_all('td')
			data['Title'] = temp[0].text
			data['Author'] = temp[8].text
			data['URL'] = None
			data['Summary'] = temp[1].text
			data['Description'] = None
			data['Keywords'] = None
			data['Size'] = None
			data['No Rows'] = temp[3].text
			data['No Columns'] = None
			data['Column Name'] = None
			data['Column Type'] = None
			data['Image dimension'] = None
			data['Number of Downloads'] = None
			data['Number of Views'] = None
			data['Anonymised Labels'] = None
			data['License'] = None
			data['File type'] = temp[4].text.replace(","," ")
			data['Algorithm'] = temp[5].text.split(",")
			data['Update'] = temp[6].text
			print data
			data_list.append(data)
	return data_list



if __name__ == '__main__':
	url = "https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research"

	try:
		page = requests.get(url)
		soup = BeautifulSoup(page.text,"html.parser")
		data = extractData(soup)
		df = pd.DataFrame(data_list)
		df.to_csv('standard.csv', encoding='utf-8')
	except Exception as e:
		print e
