
import pandas as pd
import numpy as np
from numpy.linalg import svd

def get_top_brands(matrix, columns, i):
  matrix = np.array(matrix)
  brand_ratings = [(score, brand) for (score, brand) in zip(matrix[i,], columns) if score > 0.1]
  return sorted(brand_ratings, reverse=True)[:10]






if __name__ == '__main__':
  data = pd.read_csv('user-brands.csv')
  data_matrix = pd.pivot_table(data, rows='id', cols='brand', aggfunc=len)

  data_matrix = data_matrix.fillna(0)

  user_item_matrix = np.array(data_matrix)

  k = 100

  U, S, V = svd(user_item_matrix)

  new_user_item_matrix = np.dot(np.dot( U[:,:k], np.diag(S)[:k, :k]) , V[:k,:])


  for userId in xrange(1000,1150):
    print get_top_brands(new_user_item_matrix, data_matrix.columns, userId)
    print get_top_brands(data_matrix, data_matrix.columns, userId)

