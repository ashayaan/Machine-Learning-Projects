#!/usr/bin/python

import cgi   

form = cgi.FieldStorage()   
input_text = form.getfirst("textinput", "0")
print "Content-type:text/html\r\n\r\n"
print "<html>"
print "<body>"
print "<p>%s</p>" % input_text
print "</body>"
print "</html>"