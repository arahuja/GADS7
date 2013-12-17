import oauth2 as woof
import json
import urllib2 as urllib
import train_model

model, vec = train_model.train_insult_model()

access_token_key = "<Enter your code here>"
access_token_secret = "<Enter your code here>"

consumer_key = "<Enter your code here>"
consumer_secret = "<Enter your code here>"

_debug = 0

oauth_token    = woof.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = woof.Consumer(key=consumer_key, secret=consumer_secret)

signature_method_hmac_sha1 = woof.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
def twitterreq(url, method, parameters):
  req = woof.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

  req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

  headers = req.to_header()

  if http_method == "POST":
    encoded_post_data = req.to_postdata()
  else:
    encoded_post_data = None
    url = req.to_url()

  opener = urllib.OpenerDirector()
  opener.add_handler(http_handler)
  opener.add_handler(https_handler)

  response = opener.open(url, encoded_post_data)

  return response

def fetchsamples():
  url = "https://stream.twitter.com/1/statuses/sample.json"
  parameters = {'track' : 'something'}
  response = twitterreq(url, "GET", parameters)
  for line in response:
    #Iterating over every related to topic
    text = json.loads(line.strip())
    ### Add code here ####
    ### Find tweets related to topic and then classify using the model as insult or not - print those that are insults

if __name__ == '__main__':
  fetchsamples()
