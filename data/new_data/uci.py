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
	data_list = []
	for row in batch.find_all('tr'):
		data = {} 
		count+=1
		if(count%2 != 0):
			continue
		flag = 0
		link = "https://archive.ics.uci.edu/ml/"+row.find("a",href=True).get( 'href')
		
		try:
			print
			print "Trying for " + link
			print
			page = requests.get(link)
			soup2 = BeautifulSoup(page.text,'html.parser')
		except Exception as e:
			print e
			continue

		try:
			author = soup2.find_all('p',{'class':'normal'})[19].text.replace('\r','')
		except Exception as e:
			print e
			author = None
		
		try:
			description = soup2.find_all('p',{'class':'normal'})[20].text.replace('\r','')
		except Exception as e:
			description = None

		try:
			views = soup2.find_all('p',{'class':'normal'})[18].text.replace('\r','')
		except Exception as e:
			views = None

		try:
			meta_data = row.find_all(string=lambda text:isinstance(text,Comment))
			summary = meta_data[0].split('<')[2].split('>')[1]
			keywords = meta_data[1].split('<')[2].split('>')[1]
		except Exception as e:
			summary = keywords = None
		
		temp_list = []
		for element in row.find_all('td'):
			if(flag == 0):
				flag+=1
				continue
			temp_list.append(element.text.replace(u'\xa0', u' '))
		
		# print temp_list
		data['Title'] = temp_list[1]
		data['Author'] = author
		data['URL'] = link
		data['Summary'] = summary.replace('&nbsp;', '')
		data['Description'] = description
		data['Keywords'] = keywords.replace('&nbsp;', '')
		data['Size'] = None
		data['No Rows'] = temp_list[5]
		data['No Columns'] = temp_list[6]
		data['Column Name'] = None
		data['Column Type'] = temp_list[2]
		data['Image dimension'] = None
		data['Number of Downloads'] = None
		data['Number of Views'] = views
		data['Anonymised Labels'] = None
		data['License'] = None
		data['File type'] = None
		data['Algorithm'] = temp_list[3].split(",")
		data['Update'] = temp_list[7]
		print data
		data_list.append(data)

	return data_list

if __name__ == '__main__':
	url = "https://archive.ics.uci.edu/ml/datasets.html"
	# page = requests.get(url)
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

	data_list = extractData(batch)
	df = pd.DataFrame(data_list)
	df.to_csv('uci.csv', encoding='utf-8')