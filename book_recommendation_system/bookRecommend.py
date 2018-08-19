# this py is converted from jupeter notebook
import pandas as pd

books = pd.read_csv('../../dataset/booksDataset/bxBooks.csv',
                    sep=';', error_bad_lines=False, encoding='latin-1')
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
                 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv('../../dataset/booksDataset/bxUsers.csv',
                    sep=';', error_bad_lines=False, encoding='latin-1')
users.columns = ['userID', 'location', 'age']

ratings = pd.read_csv('../../dataset/booksDataset/bookRating.csv',
                      sep=';', error_bad_lines=False, encoding='latin-1')
ratings.columns = ['userID', 'ISBN', 'bookRating']

# print(books.shape)
# print(users.shape)
# print(ratings.shape)

# in books image data is not required hence can beremoved
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)
# books.dtypes
#pd.set_option('display.max_colwidth', -1)
# books.yearOfPublication.unique()
#books.loc[books.yearOfPublication == 'Gallimard']
books.loc[books.ISBN == '2070426769', 'yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769',
          'bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769', 'publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769',
          'bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

#books.loc[books.ISBN == '2070426769']
#books.loc[books.yearOfPublication == 'DK Publishing Inc']

books.loc[books.ISBN == '078946697X',
          'bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
books.loc[books.ISBN == '078946697X', 'bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X', 'publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953',
          'bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
books.loc[books.ISBN == '0789466953', 'bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953', 'publisher'] = "DK Publishing Inc"

# convert all string in yearOfpublication to number
books.yearOfPublication = pd.to_numeric(
    books.yearOfPublication, errors='coerce')
import numpy as np
# the dataset was created in 2004 so all data before 2004 are invalid
books.loc[(books.yearOfPublication > 2004) | (
    books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
# fill null with mean values
books.yearOfPublication.fillna(
    round(books.yearOfPublication.mean()), inplace=True)
# change dataType as well
books.yearOfPublication = books.yearOfPublication.astype(np.int32)
books.loc[books.publisher.isnull()]

books.loc[(books.ISBN == '193169656X'), 'publisher'] = 'others'
books.loc[(books.ISBN == '1931696993'), 'publisher'] = 'others'
users.age = pd.to_numeric(users.age, errors='coerce')
# users below 7 and above 10 does not make sense hence can be removed
users.loc[(users.age > 90) | (users.age < 7), 'age'] = np.nan
users.age = users.age.fillna(round(users.age.mean()))
users.age = users.age.astype(np.int32)
# users.head(1)

# now we have remove unwanted data from rating
# rating should conatin data matching ISBN with book.ISBN
ratings = ratings[ratings.ISBN.isin(books.ISBN)]
# and rating should containd userId that are users csv
ratings = ratings[ratings.userID.isin(users.userID)]

# rating 0 is of no use use for us so we will remove it from users and ratings
ratings_new = ratings[ratings.bookRating != 0]
users_new = users[users.userID.isin(ratings_new.userID)]

ratings_new['bookRating'].value_counts().plot('bar')
ratings_new[ratings_new.ISBN == '0155061224']

# group by ISBN and add ratings
ratings_count = pd.DataFrame(ratings_new.groupby(['ISBN'])['bookRating'].sum())
# get top 10 books for recommendation
top10 = ratings_count.sort_values('bookRating', ascending=False).head(10)
print('top 10 recommened books!')
top10.merge(books, left_index=True, right_on='ISBN')

# continue here
# https://towardsdatascience.com/my-journey-to-building-book-recommendation-system-5ec959c41847
