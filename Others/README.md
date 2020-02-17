
# To create Table:

You can create tables with pipes | and hyphens -. Hyphens are used to create each column's header, while pipes separate each column. You must include a blank line before your table in order for it to correctly render.

			 
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

The pipes on either end of the table are optional.

Cells can vary in width and do not need to be perfectly aligned within columns. There must be at least three hyphens in each column of the header row.


You can align text to the left, right, or center of a column by including colons : to the left, right, or on both sides of the hyphens within the header row.

| Left-aligned | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |


It's very easy to make some words **bold** and other words *italic* with Markdown. You can even [link to Google!](http://google.com)

https://help.github.com/en/articles/basic-writing-and-formatting-syntax

https://help.github.com/en/articles/organizing-information-with-tables

# 
https://www.fluentu.com/blog/english/blogs-in-english/

https://www.wordstream.com/blog/ws/2014/08/07/improve-writing-skills

from pyunpack import Archive
Archive('data.7z').extractall("<output path>")

import py7zlib
f7file = "<mypath>/boost_1_60_0.7z"
with open(f7file, 'rb') as f:
     z = py7zlib.Archive7z(f)
     z.list()
     

