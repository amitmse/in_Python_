#####################################################################################################################
#####################################  NLP  #########################################################################
#####################################################################################################################

from collections import Counter
import pandas
import re

#### https://www.dataquest.io/blog/natural-language-processing-with-python/

headlines = ["PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"]

unique_words = list(set(" ".join(headlines).split(" ")))

def make_matrix(headlines, vocab):
        matrix = []
        for headline in headlines:
                # Count each word in the headline, and make a dictionary.
                counter = Counter(headline)
                # Turn the dictionary into a matrix row using the vocab.
                row = [counter.get(w, 0) for w in vocab]
                matrix.append(row)
        df = pandas.DataFrame(matrix)
        df.columns = unique_words
        return df
    
print(make_matrix(headlines, unique_words))
new_headlines = [re.sub(r'[^\w\s\d]','',h.lower()) for h in headlines]
new_headlines = [re.sub("\s+", " ", h) for h in new_headlines]
unique_words = list(set(" ".join(new_headlines).split(" ")))
print(make_matrix(new_headlines, unique_words))

from nltk import word_tokenize
sentence = "What is the weather in Chicago?"
tokens = word_tokenize(sentence)
print tokens

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
clean_tokens = [w for w in tokens if not w in stop_words]
print clean_tokens

import nltk
tagged = nltk.pos_tag(clean_tokens)
print tagged
'''
[('What', 'WP'), ('weather', 'NN'), ('Chicago', 'NNP'), ('?', '.')]
         'WP'  - WH-pronoun
         'NNP' - Proper noun, singular
         'NN'  - noun, singular
         '.'   - Punctuation marks
'''
print(nltk.ne_chunk(tagged))

#############################################################################################################
#### https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
import re 
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
from nltk import word_tokenize, pos_tag

noise_list = ["is", "a", "this", "..."]
def _remove_noise(input_text):
		words 				= input_text.split() 
		noise_free_words 	= [word for word in words if word not in noise_list] 
		noise_free_text 	= " ".join(noise_free_words)
		return noise_free_text
_remove_noise("this is a sample text")
 


def _remove_regex(input_text, regex_pattern):
		urls = re.finditer(regex_pattern, input_text)
		for i in urls:
				input_text = re.sub(i.group().strip(), '', input_text)
		return input_text
regex_pattern = "#[\w]*" 
_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)


lem 	= WordNetLemmatizer()
stem 	= PorterStemmer()
word 	= "multiplying" 
lem.lemmatize(word, "v")
stem.stem(word)


lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}
def _lookup_words(input_text):
		words = input_text.split() 
		new_words = [] 
		for word in words:
				if word.lower() in lookup_dict:
						word = lookup_dict[word.lower()]
				new_words.append(word) 
				new_text = " ".join(new_words) 
				return new_text
_lookup_words("RT this is a retweeted tweet by Shivam Bansal")


text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print pos_tag(tokens)



doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]



import gensim from gensim
import corpora
# Creating the term dictionary of our corpus, where every unique term is assigned an index.  
dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
# Results 
print(ldamodel.print_topics())



def generate_ngrams(text, n):
		words = text.split()
		output = []  
		for i in range(len(words)-n+1):
				output.append(words[i:i+n])
		return output
generate_ngrams('this is a sample text', 2)



from sklearn.feature_extraction.text import TfidfVectorizer
obj = TfidfVectorizer()
corpus = ['This is sample document.', 'another random document.', 'third sample document text']
X = obj.fit_transform(corpus)
print X

#############################################################################################################

#https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/

import nltk  nltk.download()

#####Noise Removal
# Sample code to remove noisy words from a text
noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
		words = input_text.split() 
		noise_free_words = [word for word in words if word not in noise_list] 
		noise_free_text = " ".join(noise_free_words) 
		return noise_free_text
_remove_noise("this is a sample text")

# Sample code to remove a regex pattern 
import re 

def _remove_regex(input_text, regex_pattern):
		urls = re.finditer(regex_pattern, input_text) 
		for i in urls: 
			input_text = re.sub(i.group().strip(), '', input_text)
		return input_text
regex_pattern = "#[\w]*"
_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)


########Lexicon Normalization
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")
stem.stem(word)

########Object Standardization
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love", "..."}
def _lookup_words(input_text):
		words = input_text.split() 
		new_words = [] 
		for word in words:
			if word.lower() in lookup_dict:
				word = lookup_dict[word.lower()]
			new_words.append(word) new_text = " ".join(new_words) 
			return new_text
_lookup_words("RT this is a retweeted tweet by Shivam Bansal")

#######Part of speech tagging
from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print pos_tag(tokens)

#######

#############################################################################################################

from nltk import word_tokenize
tokens = word_tokenize(sentence)
tokens
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words
clean_tokens = [w for w in tokens if not w in stop_words]
clean_tokens
import nltk
tagged = nltk.pos_tag(clean_tokens)
tagged
print(nltk.ne_chunk(tagged))
import urllib
response = urllib.urlopen(url)
raw = response.read().decode('utf8')
raw[:75]
len(raw)
tokens = word_tokenize(raw)
len(tokens)
tokens[:10]
text = nltk.Text(tokens)
text[1024:1062]
print text[1024:1062]
text.collocations()
raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")
raw.find("PART I")
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urllib.urlopen(url).read().decode('utf8')
html[:60]
from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
len(llog.entries)
post = llog.entries[2]
post.title
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
' '.join(silly)
';'.join(silly)
''.join(silly)
empty = []
nested = [empty, empty, empty]
nested
nested[1]
nested[1].append('Python')
nested[1]
nested

#############################################################################################################

