from collections import defaultdict
import operator

def jaccard_distance(l1, l2):
    union = set(l1).union(set(l2))
    intersection = set(l1).intersection(set(l2))

    return len(intersection) / float(len(union))

# Load data
f = open('user-brands.csv')
brand_users = defaultdict(list)  # Given a brand, which users are followers
user_brands = defaultdict(list)  # Given a user, which brands does the user follow
for line in f:
    user, brand = line.strip().split(',', 1)
    brand_users[brand].append(user)
    user_brands[user].append(brand)

# Create similarity "matrix"
#similarity = defaultdict(dict)  # Given two brands, what is the similarity score using Jaccard coefficient
similarity = {}
brand_list = brand_users.keys()
for brand1, users1 in brand_users.items():
    for brand2, users2 in brand_users.items():
        key = tuple(sorted((brand1, brand2)))
        if not key in similarity:    
            similarity[key] = jaccard_distance(users1, users2)

def get_similar_brands(brand, threshold = 0):
    """Given a brand, return similar brands with scores"""
    brand_scores = defaultdict(int)
    ## TODO : Implement
    ## For all other brands
    for other_brand in brand_list:
        if other_brand != brand:
            #Set the as above
            key = tuple(sorted((brand, other_brand)))

            #Look up sim score for the pair
            sim = similarity[key]

            # If it passes the threshold
            if sim > threshold:
                #add it to the dict
                brand_scores[other_brand] = sim

    ## Get the brands that have similarity > threshold
    # return key value pairs, where the key is brand and the value is score
    return brand_scores.items()
    ## Return a list of (brand, similarity_score) pairs



def get_brand_recommendations(user):
    """Given a user, return recommended brands with scores"""
    all_brand_scores = defaultdict(int)

    for other_brand in brand_list:
        all_brand_scores[other_brand] = sum ( [similarity[ tuple(sorted((brand, other_brand)))] for brand in user_brands[user]])
        # for brand in user_brands[user]:
        #     key = tuple(sorted((brand, other_brand)))
        #     sum += similarity[key]   
        # all_brand_scores[other_brand] = sum 
    ##TODO: Implement
    ## Take the brands that this user likes
    ## Find similar brands to each brand this user likes
    ## Get total similarity to all other brands based on similarirt to each brand this user likes
    return sorted([(score, key) for (key, score) in all_brand_scores.items()], reverse=True)[:10]
    ## Return the top scoring brands by total similarity

#user = '90217'
#user = '89112'
user = '89116'
print "Current brands: {}".format(user_brands.get(user))
print "Recommendations:"
print get_brand_recommendations(user)


"""Optimizations:
1. Limit similarity scores to brands with at least __ followers
2. Filter recommendations via a score threshold
"""
