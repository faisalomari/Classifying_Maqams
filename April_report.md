We've implemented the model as described in the article we are working with, we've trained the model on the dataset we have, and that worked!
But the problem we had that the train we've done didn't include the parameters that have the best results and that is due to the GPU memory limit.
We are working currently on Faisal's private GPU which is RTX3060ti with 8GB, but according to the code, the required memory needed to run
the model according to the mentioned parameters in the article is about 30GB. the way we ran the model is with chhaning the parameters of the CNN layers,
in addition to changing the batch_size, the loss we had is about 1%, but when we've tested the model it gave an accuracy of about 30-50%.

The plan for the next month is as following:
1) to find a way to have more GPU memory, maybe we'll move to Google Colab (maybe pro)...
2) Run the model and make sure that the accuarcy is the best we can have.
3) Continue implementing the model, we have to implement LSTM and CNN+LSTM additional models and pass the data on these models. 
4) Using the MFCC features as mentioned in the article.

Additional notes:
We've noticed that the dataset we have is not the best in quality, so we've discussed the idea of building new dataset, with better quality,
and better Maqam accuarcy in each audio sample.
I (Faisal) have searched about the idea, and I saw that it is not really hard to create a new dataset, there is a lot of long (2+ hours)
of videos in youtube that have a recitation on the same maqam during the whole video time.
The purpose of creating a new dataset is as we've mentioned to have a better quality and accuarcy, in addition that we think that the dataset we have is not really big,
and thats becuase it didn't really take a lot of time training the dataset, so if we had enough time in the end of the training, maybe we will do this step.
Having more data will lead to better accuarcy and stronger model results.
