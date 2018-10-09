import numpy as np
import pandas

# recommeding to most popularly used songs in dataset
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations

#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    # initailize data
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
   
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Use the item similarity based recommender system model to make recommendations
    def recommend(self, user):
        #A. Get all unique songs for this user
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))

        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C. Construct item cooccurence matrix of size 
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
       
        #D. Use the cooccurence matrix to make recommendations
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs. (column sum)
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score (sort based on index)
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
        
    #Construct cooccurence matrix
    '''
   | ---------------------------------All songs---------------------------------------------
   | songs|    1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10  |
   | ---------------------------------------------------------------------------------------
   u   1  |   0    |   0   |   11  |   23  |   0   |   44  |   8   |   11  |   0   |   0   |
   s   2  |   0    |   4   |   1   |   3   |   0   |   44  |   8   |   11  |   0   |   0   |
   e   3  |   10   |   0   |   0   |   2   |   0   |   44  |   8   |   11  |   0   |   0   |
   r   4  |   11   |   33  |   0   |   23  |   0   |   44  |   8   |   11  |   0   |   0   |
   s   5  |   1    |   0   |   0   |   1   |   0   |   44  |   8   |   11  |   0   |   0   |
   o   6  |   0    |   0   |   0   |   23  |   0   |   44  |   8   |   11  |   0   |   0   |
   n   7  |   0    |   22  |   0   |   5   |   0   |   44  |   8   |   11  |   0   |   0   |
   g   8  |   11   |   0   |   0   |   7   |   0   |   44  |   8   |   11  |   0   |   0   |
   
    '''
    # https://blogs.msdn.microsoft.com/carlnol/2012/06/23/co-occurrence-approach-to-an-item-based-recommender/
    # user_songs = list of unique user songs
    # all_songs = list of all unique songs  form train data
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        #A. Get users for all songs in user_songs.
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))  # array of dict(unique users for user song i)
   
        #Initialize the item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_songs)):     # all unique songs in data 
            if(i%50 == 0):
                print('training data for song:', i)
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())        # dict of unique user id listening song i
            
            for j in range(0,len(user_songs)):   # unique songs for this user     
                    
                #Get unique listeners (users) of user song (item) j
                users_j = user_songs_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                # interserction of users listening song i form all songs and users listening song j form user_songs
                users_intersection = users_i.intersection(users_j)
                
                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                # The Jaccard coefficient measures similarity between finite sample sets
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        print('-----------', cooccurence_matrix)
        return cooccurence_matrix

    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
       
        return all_items
    
        #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C. Construct item cooccurence matrix of size 
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #D. Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations

        
        