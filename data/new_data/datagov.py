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


def extract_data(soup):
	
	for element in soup.find_all("li",{"class":"dataset-item has-organization"}):
		data = {}
		data["Anonymised Labels "] = None
		data["Title"] = element.find("a", href=True).text.strip()
		data["Url"] = "https://catalog.data.gov"+ element.find("a",  href=True).get('href')
		data["Number of views"] = int(element.find("span",{"class":"recent-views"}).get("title").split()[0])
		data["Number of Downloads"] = int(element.find("span",{"class":"recent-views"}).get("title").split()[0])
		data["Author"] = element.find("p",{"class":"dataset-organization"}).text.replace("\xe2\x80\x94","").strip()
		data["Summary"] = None
		try:
			data["File Type"] =  element.find("ul",{"class":"dataset-resources unstyled"}).find("a").text.strip()
		except Exception as e:
			data["File Type"]= None
		data["No Rows"] = None
		data["No Columns"] = None
		data["Column Name"] = None
		data["Column Type"] = None
		data["Size"] = None
		data["Image Dimension"] = None
		date = None

		try:
			print "Visting " + data["Url"]
			page2 = requests.get(data["Url"])
			soup2 = BeautifulSoup(page2.text,"html.parser")
			data["Description"] = soup2.find("div",{"class":"notes embedded-content"}).text.replace("\n"," ").replace("\r"," ").replace("\t"," ").replace(",","").replace(":","").strip()
			tags = []
			for t in soup2.find("ul",{"class":"tag-list well"}).find_all("li"):
				tags.append(t.text.strip())
			data["Keywords"] = tags
			license = soup2.find("div",{"access-use-main"}).find_all("span")[-1].text
			license = license.split(":")[1]
			license = license.strip("\n").strip("\r").strip()
			data["License"] = license
			metadata = soup2.find("section",{"class":"module-content additional-info"}).find_all("tr")
			for d in metadata:
				if (d.find("th",{"class","dataset-label"}).text == "Data Last Modified"):
							data["Updated"] = d.find("td",{"class","dataset-details"}).text.strip()
		except Exception as e:
			print e
			continue
		
		print
		print data
		print
		data_list.append(data)
		

if __name__ == '__main__':
	base_url = "https://catalog.data.gov/dataset?page="
	for count in range(91,100):
		url = base_url + str(count)
		try:
			print "Trying " + url
			page = requests.get(url)
			soup = BeautifulSoup(page.text,"html.parser")
			extract_data(soup)
			df = pd.DataFrame(data_list)
			df.to_csv('test.csv', encoding='utf-8')
		except Exception as e:
			print e
		