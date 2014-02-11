from collections import defaultdict
import operator

def jaccard_distance(l1, l2):
    ##TODO:Implement write jaccard distance function

# Load data
f = open('user-brands.csv')
brand_users = defaultdict(list)  # Given a brand, which users are followers
user_brands = defaultdict(list)  # Given a user, which brands does the user follow
for line in f:
    user, brand = line.strip().split(',', 1)
    brand_users[brand].append(user)
    user_brands[user].append(brand)

# Create similarity "matrix"
similarity = {}  # Given two brands, what is the similarity score using Jaccard coefficient
brand_list = brand_users.keys()
for brand1, users1 in brand_users.items():
    for brand2, users2 in brand_users.items():
        ##TODO:Implement compute similarity between rbands

def get_similar_brands(brand, threshold):
    """Given a brand, return similar brands with scores"""
    brand_scores = defaultdict(int)
    ## TODO : Implement
    ## For all other brands
    ## Get the brands that have similarity > threshold


    ## Return a list of (brand, similarity_score) pairs



def get_brand_recommendations(user):
    """Given a user, return recommended brands with scores"""
    all_brand_scores = defaultdict(int)
    ##TODO: Implement
    ## Take the brands that this user likes
    ## Find similar brands to each brand this user likes
    ## Get total similarity to all other brands based on similarirt to each brand this user likes

    ## Return the top scoring brands by total similarity

user = '90217'
# user = '89112'
# user = '89116'
print "Current brands: {}".format(user_brands.get(user))
print "Recommendations:"
print get_brand_recommendations(user)


"""Optimizations:
1. Limit similarity scores to brands with at least __ followers
2. Filter recommendations via a score threshold
"""
