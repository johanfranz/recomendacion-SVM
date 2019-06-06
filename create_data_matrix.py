
import csv
import numpy as np
import pandas as pd
import pickle

#we're looking to populate these 4 dictionaries
movieId_movieName = {}
movieId_movieCol  = {}
userId_userRow    = {}
userId_rating     = {}

#for reference
movieId_isRated   = {}

#then populate a data matrix based on the userId_rating dictionary


'''
Read Basic Movie Info
'''

path     = 'data'
filename = 'movies.csv'


dataFrame = pd.read_csv('{}/{}'.format(path,filename))

for i in range(len(dataFrame['movieId'])):
    movieId = dataFrame['movieId'][i]
    
    movieId_movieName[movieId] = dataFrame['title'][i]    
    movieId_isRated[movieId]    = 0

   
'''
Read in the ratings
'''
path     = 'data'
filename = 'ratings.csv'


dataFrame = pd.read_csv('{}/{}'.format(path,filename))

for i in range(len(dataFrame)):
    userId  = dataFrame['userId'][i]
    movieId = dataFrame['movieId'][i]
    rating  = dataFrame['rating'][i]
    
    if userId not in userId_rating.keys():
        userId_rating[userId] = [(movieId, rating)]
    else:
        userId_rating[userId].append((movieId, rating))
    
    movieId_isRated[movieId] = 1
        
print(userId_rating[1])
    

for movieId, isRated in movieId_isRated.items():
    if isRated == 0:
        del movieId_movieName[movieId]

'''
Create the row-column data

the rows and columns will have the entries for Ids in sorted order.
Therefore:
    the ith row    in the data matrix will be the ith key in sorted movieIds
    the jth column in the data matrix will be the jth key in sorted userIds
'''
userId_userRow
movieId_movieCol

i = 0
for movieId in sorted(movieId_movieName):
    movieId_movieCol[movieId] = i
    i+=1

i=0
for userId in sorted(userId_rating):
    userId_userRow[userId] = i
    i+=1

m = len(userId_userRow.keys())
n = len(movieId_movieCol.keys())
A = np.zeros((m,n))


    
print(A.shape)
for userId, ratings in userId_rating.items():
    for rating in ratings:
        movieId   = rating[0]
        score     = rating[1]
        
        if (userId in userId_userRow and movieId in movieId_movieCol):
            i = userId_userRow[userId]

            j = movieId_movieCol[movieId]
            A[i,j] = score

ratingCount = 0
for i in range(m):
    for j in range(n):
        if (A[i][j] != 0):
#             if(ratingCount < 20):
#                 print(A[i][j])
            ratingCount += 1


print('Number of ratings = {}'.format(ratingCount))
print('Total entries = {}'.format(m*n))
print('Sparsity = {}%'.format(ratingCount*100/(m*n)))


'''
write the relevant dictionaries and matrix to a file
'''
# movieId_movieName
# movieId_movieCol
# userId_userRow
# userId_rating
# A

d = {'movieId_movieName': movieId_movieName,
     'movieId_movieCol' : movieId_movieCol,
     'userId_userRow'   : userId_userRow,
     'userId_rating'    : userId_rating }
pickle.dump(A, open('data/data_matrix.p', 'wb'))
pickle.dump(d, open('data/data_dicts.p', 'wb'))
print (A.shape)