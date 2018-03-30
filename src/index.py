#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

import pandas as pd

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir, analyzer):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer)
        ticker = Ticker()
        print 'commit index',
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print 'done'

    def indexDocs(self, root, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setStoreTermVectors(True)
        t2.setStoreTermVectorOffsets(True)
        t2.setStoreTermVectorPositions(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)

	file_path = root+'all.csv'
	#fd = open(file_path)
	df = pd.read_csv(file_path)
	rows = df.shape[0]
	title = df['Title']
	summary = df['Summary']
	desc = df['Description']
	keywords = df['Keywords']
	#fd.close()
	#contents_list = [x.strip() for x in contents]
	for i in xrange(rows):
		try:
			titleI = title[i]
			summaryI = summary[i]
			descI = desc[i]
			keywordsI = keywords[i]
			doc = Document()
			#doc.add(Field("id", str(i), t1))
			doc.add(Field("title", titleI, t2))
			doc.add(Field("summary", summaryI, t2))
			doc.add(Field("description", descI, t2))
			doc.add(Field("keywords", keywordsI, t2))
			writer.addDocument(doc)
		except Exception, e:
			print "Failed in indexDocs:", e
		
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print IndexFiles.__doc__
        sys.exit(1)
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    start = datetime.now()
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        IndexFiles(sys.argv[1], os.path.join(base_dir, INDEX_DIR),
                   StandardAnalyzer())
        end = datetime.now()
        print end - start
    except Exception, e:
        print "Failed: ", e
        raise e
