
# Music Genre Predictor

This is a code to predict the type of Music Genre a person is interested considering the inputs such as 'age' and 'gender'.


## Overview : 

 - Checking the accuracy of the model.
 - Saving the trained model for the future use.
 - Making the predictions using the saved model.


### Checking the accuracy of the model

Import the required libraries such as 'pandas' and 'sklearn'. Initially the data is loaded from csv file and split into x(inputs from the user) and y(result to be predicted).

Next, the train_test_split function from scikit-learn's model_selection module is used to split the x and y variables into training and testing sets. The test set will be 20% of the data and the rest will be the training set.

Then the accuracy of the model is tested as in to find it desirable for the most relevant predictions,if not the machine has to be trained for more data. The usual accuracy to be expected is more than 70% in most of the cases.

![Screenshot from 2022-12-30 14-14-39](https://user-images.githubusercontent.com/116273227/210052033-33fc6945-3b1d-402f-93c0-a9d1ae344a49.png)

```bash
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('/home/shridhar/Jupyter/music.csv')
x = music_data.drop(columns = ['genre'])
y = music_data['genre']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

score = accuracy_score(y_test,predictions)
print(f'{int(score*100)}%')
```
### Saving the trained model for the future use

Now that our model is trained and ready for use it has to be saved so that each time we want to predict is must not do the above steps.so the model is saved as 'music_recmmnd.joblib'

```bash
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)

model = joblib.dump(model , 'music_recmmnd.joblib')
```

### Making the predictions using the saved model

```bash
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

model = joblib.load('/home/shridhar/Jupyter/music_recmmnd.joblib')

# sample prediction set
prediction = model.predict([[21,1]])
```
### Output : 

![Screenshot from 2022-12-30 14-25-03](https://user-images.githubusercontent.com/116273227/210052459-7a235e33-f530-4cee-94ad-fbd8b2fdbcdc.png)


## 🚀 About Me
I'm a passionate learner interested in exploring new technologies, Do reach me out for any projects and collabarations.

  Thank you !

