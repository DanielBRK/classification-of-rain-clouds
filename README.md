# Final Project - Le Wagon Berlin - Batch 1014 - Classification of rain clouds

Hello :) we are Sophie, Abigail, Tugce and Daniel. During our project week at the Le Wagon Bootcamp
we built an app, which allows you to take a picture of sky with the camera of your device, and the app will tell you, if there is a rain cloud near you. We achived that with a convolutional neural network and a transfer learning approach using the resnet50v2 model. For training the model we used weather data and sky images provided from the website http://nw3weather.co.uk/ (Thanks to the website for allowing us to use the data!)
In this repository you find the python code for the models (folders cnnbinary & cnncategorical) and the python code for the API/user-interface (folder userinterface). You can run the models with the scripts runcnnbinary.py & runcnncategorical.py:

```
python runcnnbinary.py
python runcnncategorical.py
```

the models need the training data. you can download the training data and final trained models (for comparison) by using the script downloaddata.py:

```
python downloaddata.py
```

When you want to run run the API and user-interface, please use the commands:

```
streamlit run userinterface/main.py
uvicorn userinterface.app:app --port 8080 --reload
```

You need the following libraries and packages for using the code:

- streamlit
- PIL
- requests
- streamlit_cropper (see https://github.com/turner-anderson/streamlit-cropper thanks to the author:)
- numpy
- time
- fastapi
- uvicorn
- tensorflow & keras
- matplotlib

Have fun with the app and the models :)

Sophie, Abigail, Tugce and Daniel
