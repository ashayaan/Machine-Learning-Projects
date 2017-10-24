import sys
from bs4 import BeautifulSoup
import requests
import re
import json
import urllib2

def get_url(url, pattern, out):
#url = sys.argv[1]
	url = url.strip()
        r  = requests.get(url)
        data = r.text
        soup = BeautifulSoup(data,"lxml")
        for link in soup.find_all('a'):
                lst = link.get('href')
                if(lst):
                        if(lst.find(pattern) == 0 and lst.find('/html') == -1 and lst.find('/original') == -1):
				json_url = 'https://catalog.data.gov'+lst
				response = urllib2.urlopen(json_url)
				info = response.info()
    				if (info.subtype == "json"):     
					data=json.load(response)
					access = data["accessLevel"]
					desc = data["description"].replace(',',' ').replace('\n',' ').replace('\r',' ')
					distributions = data["distribution"]
					mediaTypes = ""
					for i in range(len(distributions)):
						if "mediaType" in distributions[i]:
							mediaTypes= mediaTypes + ';' + distributions[i]["mediaType"]
					if 'keyword' in data:
						keywords = data["keyword"]
					else:
						keywords = ['N/A']
					title = data["title"].strip()
					source = data["publisher"]["name"].replace(',',' ').strip()
					print url
					out_str =  (title +","+source+","+url+ "," + desc + "," + ';'.join(str(k).strip() for k in keywords)+'\n').encode('utf-8')
					out.write(out_str)
        return 0


def get_attribute_webpage(url_file_name,pattern,out):
	url_file = open(url_file_name, 'r')
	for line in url_file:
		get_url(line, pattern,out)
	

if __name__ == '__main__':
	out = open('datagov.csv','a')
	#out.write(("URL"+";"+"Title" + ";"+ "Description" + ";" + "Keywords\n").encode('utf-8'))
	get_attribute_webpage(sys.argv[1],sys.argv[2],out)
	out.close()
