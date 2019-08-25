----------Group Information-----
PROJECT:
FRIENDLY CHATBOT
MEMBERS:
	Abhinav Jindal
	Mattela Nithish
	Shailendra kumar Gupta
	Shreyanshu Shekhar

----------How to Run------------
->To run the normal model
	$python run.py <modes>
	where <mode>='trainmode' (for training),
			  ='chatmode' (for chatting)


------------Requirements---------
tensorflow 1.13.1
python3
numpy
keras
sklearn

------------Packet structure------

|
|--saved_data
|
|--dataset
    |
|   --cornell_movie_dialogs_corpus
|
|--code
    |
    --run.py and many more.
    
-------------About files----------
run.py -> This file is to run the model either in train or chat the model.
preprocessed data is stored in saved_data in saved_data folder, file name is data.pkl.
RL results are not available due to some training error and time constraints.
----------------------------------

