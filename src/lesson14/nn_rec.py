from collections import defaultdict
import operator

def jaccard_distance(first_set, second_set):
    return len(set(first_set).intersection(set(second_set))) * 1.0 / len(set(first_set).union(set(second_set)))


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
        if brand1 != brand2:
            key = tuple(sorted([brand1, brand2]))  # key is tuple of brands, sorted alphabetically
            sim = jaccard_distance(users1, users2)
            similarity[key] = sim

# List all similarity scores
# print sorted(similarity.iteritems(), key=operator.itemgetter(1))


def get_similar_brands(brand):
    """Given a brand, return similar brands with scores"""
    brand_scores = defaultdict(int)
    for other_brand in brand_users.keys():
        if brand == other_brand:
            continue
        key = tuple(sorted([brand, other_brand]))
        sim = similarity.get(key, 0)
        if sim > 0:
            brand_scores[other_brand] += sim
    return brand_scores


def get_brand_recommendations(user):
    """Given a user, return recommended brands with scores"""
    all_brand_scores = defaultdict(int)
    for brand in user_brands[user]:
        brand_scores = get_similar_brands(brand)
        for brand1, score in brand_scores.items():
            if brand1 not in user_brands[user]:
                all_brand_scores[brand1] += score
    return sorted(all_brand_scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]

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
