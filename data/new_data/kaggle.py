from bs4 import BeautifulSoup
import csv
import sys
import pandas as pd
import time
import dryscrape
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
	source_code = open('kaggle.html', 'r').read()
	soup = BeautifulSoup(source_code, "html.parser")
	session = dryscrape.Session()

	count = 0
	data_list = []
	for block in soup.find_all("div",{"class":"block-link block-link--bordered"}):
		count +=1
		if(count < 1037):
			print count
			continue
		data = {}

		data["Title"] = block.find("div",{"class":"dataset-item__main-title"}).text
		data["Author"] = block.find("span",{"class":"dataset-item__main-owner"}).text
		data["URL"] = "https://www.kaggle.com" + block.find("a", {"class" : "dataset-item__main-title-link dataset-item__link"}, href=True).get('href')
		link = data["URL"]
		data["Summary"] = block.find("div",{"class":"dataset-item__main-subtitle"}).text
		
		#variable initialization
		description=None
		downloads = 0
		views = 0
		column_name = []
		column_type = []

		#Visiting Individual pages to extract the description, number of downloads and views for the dataset
		try:
			print "Trying for description " + link
			session.visit(link)
			response = session.body()
			soup2 = BeautifulSoup(response,"html.parser")
			description = soup2.find("div",{"class":"markdown-converter__text--rendered"}).text
			unordered_list = soup2.find_all("li",{"class":"horizontal-list-item horizontal-list-item--bullet horizontal-list-item--default"})
			views = int(unordered_list[0].text.replace(",","").split()[0])
			downloads = int(unordered_list[1].text.replace(",","").split()[0])
			# session.render("test.png")
			session.reset()
			description = description.replace(',','')
			description = description.replace('\n',' ')
			description = description.replace('\r',' ')
			description = description.replace('\t',' ')
			description = description.replace(':','')
			# print description
		except Exception as e:
			print e
			print link + " Link doesn't Not Exists"
		
		# Extracting the tags
		tags = []
		for t in block.find_all("a",{"class":"dataset-item__tag dataset-item__link"}):
			tags.append(t.text)
		
		#Extracting the metadata
		try:
			print "Visiting " + link +"/data"
			session.visit(link+"/data")
			time.sleep(2)
			response = session.body()
			soup3 = BeautifulSoup(response,"html.parser");
			session.reset()
			for cname in soup3.find_all("div",{"class","data-preview__table-info-tooltip-name"}):
				column_name.append(cname.text)
				# print cname.text

			for ctype in soup3.find_all("div",{"class":"data-preview__table-info-tooltip-type"}):
				column_type.append(ctype.text)

		except Exception as e:
			column_name = None
			column_type = None
			print link+"/data"+" Link Doesn't Exists "
			print e

		# # Metadata of the file
		file_type =  block.find("div",{"class":"dataset-item__meta-type"}).text
		license = block.find("div",{"class":"dataset-item__meta-license"}).text
		size = block.find("div",{"class":"dataset-item__meta-size"}).text
		size,unit = size.split()
		size = float(size)
		if unit == "TB":
			size = size*1024.0*1024.0
		elif unit == "GB":
			size = size * 1024.0
		elif unit == "KB":
			size = size/1024.0

		# when the data was updated
		update = block.find("span",{"class":"dataset-item__main-update"}).find("span").get("title")[0:15]

		data["Description"] = description
		data["Keywords"] = tags
		data["Size"] = size
		data["No Rows"] = None
		data["No Columns"] = None
		data["Column Name"] = column_name
		data["Column Type"] = column_type
		data["Image Dimension"] = None
		data["Number of Downloads"] = downloads
		data["Number of views"] = views
		data["Anonymised Labels "] = None
		data["License"] = license
		data["File Type"] = file_type
		data["Updated"] = update

		# print column_name
		print 
		print data
		data_list.append(data)
		df = pd.DataFrame(data_list)
		df.to_csv('kaggle.csv', encoding='utf-8')
		print



		
