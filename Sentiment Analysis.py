import nltk
from textblob import TextBlob
from newspaper import Article
 
url = 'https://en.wikipedia.org/wiki/Mathematics'

article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary

blob = TextBlob(text)

sentiment_maths = blob.sentiment.polarity

print(sentiment_maths)

url_2 = 'https://decrypt.co/96419/defi-blue-chip-aave-mounts-rise'

article_2 = Article(url_2)

article_2.download()
article_2.parse()
article_2.nlp()

text_2 = article_2.summary

blob = TextBlob(text_2)

sentiment_crypto = blob.sentiment.polarity

print(sentiment_crypto)
