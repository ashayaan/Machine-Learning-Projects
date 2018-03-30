#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.complexPhrase import ComplexPhraseQueryParser
from org.apache.lucene.queryparser.ext import ExtendableQueryParser
from org.apache.lucene.search import BooleanClause 
from org.apache.lucene.queries import CustomScoreQuery
from org.apache.lucene.queries import CustomScoreProvider
from org.apache.lucene.util import Version
#from org.apache.lucene.index import AtomicReader

	
def run(searcher, analyzer):
	while True:
		print
		print "Hit enter with no input to quit."
		command = raw_input("Query:")
		if command == '':
			return

		print
		print "Searching for:", command
		fields = ("description", "title", "summary", "keywords")
		parser = MultiFieldQueryParser(fields, analyzer)
		query = MultiFieldQueryParser.parse(parser, command)
		#query = QueryParser("description", analyzer).parse(command)
		#query = parser.parse(command)
		scoreDocs = searcher.search(query, 50).scoreDocs
		print "%s total matching documents." % len(scoreDocs)

		for scoreDoc in scoreDocs:
			doc = searcher.doc(scoreDoc.doc)
			print 'Topic:', doc.get("title"), 'Score:', scoreDoc.score


if __name__ == '__main__':
	lucene.initVM(vmargs=['-Djava.awt.headless=true'])
	print 'lucene', lucene.VERSION
	base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
	searcher = IndexSearcher(DirectoryReader.open(directory))
	analyzer = StandardAnalyzer()
	run(searcher, analyzer)
	del searcher
