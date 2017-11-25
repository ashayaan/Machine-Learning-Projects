from bs4 import BeautifulSoup
import requests
import re
import sys

def get_dataset_url(url, pattern):
#url = sys.argv[1]
	r  = requests.get("http://" +url)
	data = r.text
	soup = BeautifulSoup(data,"lxml")
	for link in soup.find_all('a'):
    		lst = link.get('href')
		if(lst):
    			if(lst.find(pattern) == 0 and lst.find('/resource/') == -1):
				print "http://catalog.data.gov"+lst
	return 0	

if __name__ == "__main__":
	if(sys.argv[1] == "catalog.data.gov/dataset"):
		for x in range(50):
			get_dataset_url(sys.argv[1]+"?"+"page="+str(x), sys.argv[2])
	else:
		get_dataset_url(sys.argv[1], sys.argv[2])
