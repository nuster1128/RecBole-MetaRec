INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file book-crossing.inter comprising the ratings of users over the books.
Each record/line in the file has the following fields: user_id, item_id, rating

user_id: the id of the users and its type is token. 
item_id: the id of the books and its type is token.
rating: the rating of the users over the books, and its type is float.

BOOKS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file book-crossing.item comprising the attributes of the books.
Each record/line in the file has the following fields: item_id, book_title, book_author, publication_year, publisher
 
item_id: the id of the movies and its type is token.
book_title: the title of the books, and its type is token_seq.
book_author: the author of the books, and its type is token_seq.
publication_year: the year when books were published, and its type is float.
publisher: the publishers of the books, and its type is token_seq.


USERS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file book-crossing.user comprising the attributes of the users.
Each record/line in the file has the following fields: user_id, location, age
 
user_id: the id of the users and its type is token.
location: the location of the users, and its type is token_seq.
age: the age of the users, and its type is float.
