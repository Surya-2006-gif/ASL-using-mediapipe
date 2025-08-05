# ASL RECOGNIZER

The Aim of this gitub repo is to provide a  kickstarter project in computer vision.


- So,I have implemented two models in this repo

      - One of them uses only the  convolutional neural network

      - while the other one uses both ANN and CNN to combinedly make decision
  

# `model.py`  

   #### THE BASIC PIPELINE

             Detect keypoints(2d coordinates) using mediapipe

                           |
                           |
                           |
                          \ /

             Create a boundingbox around implemented

                           | 
                           |
                           |
                          \ /

             Pad around the detected bounding box

                           |
                           |
                           |
                          \ /

             Train using cnn,from the padded image                          


  - As I mentioned earlier,this model solely takes decision based on the images

  - This model overall performed very well...


# `model_with_keypoints.py`  

   #### THE BASIC PIPELINE

             Detect keypoints(2d coordinates) using mediapipe

                           |
                           |
                           |
                          \ /

             Create a boundingbox around implemented

                           | 
                           |
                           |
                          \ /

             Pad around the detected bounding box

                           |
                           |
                           |
                          \ /

        Measure the finger tip coordinates w.r.t to wrist and normalize those with the bounding box area

                           |
                           |
                           |
                          \ /
                           

    train ANN with those extracted keypoint features and train cnn with those images                  

                                        

  - This model learns how much of weights to assign to keypoints and to the image

  - Eventhough the model achieved an accuracy of 99+ in both training data and validation_data and a val_loss of less than 5.9 it didnt perform well in real time

  - I request you to fine tune the parameters
 
  
