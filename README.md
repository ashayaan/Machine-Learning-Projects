# Machine-Learning-Projects
Dataset Search Engine, predicts dataset based on the keyword entered by the user.

### Requirements
The requirements to run the application are Pandas, Numpy, scikit-learn, NLTK and Flask.

### Necessary Installation
pip install bs4 <br />
pip install Flask <br />
pip install -U scikit-learn <br />
pip install -U nltk

### Folders

The application is divided into two folders - 
* Src Folder - This folder contains all the source code for the application.
	* Models Folder - This folder contains the trained and saved models.
	* Portal Folder - This folder contains the server code and the HTML templates.
* Data Folder - This folder contains all the data files for the application.

### Running the application

To run the code unzip the folder containing all the files for the application

* Navigate to the /src/portal directory
* Run the following command: **python search_engine.py**. After the execution of the command the server starts running at the following IP address:http://127.0.0.1:5000/
* Enter the search keyword in the form. The search keyword can be one of the nodes in the ontology. For example, Science, crime, disasters, video games.
* Click on search and the results of the search engine will be displayed in a table format.


