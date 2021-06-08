from flask import Flask,render_template
from flask import request
import pandas as pd
import pickle
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
loaded_model = pickle.load(open('model.pkl','rb'))
news= pd.read_csv("train.csv")
x=news['title']
y=news['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)

def fake_news_det(fnews):
    tfid_x_train= tfvect.fit_transform(x_train.values.astype('U'))
    tfid_x_test= tfvect.transform(x_test.values.astype('U'))
    #tfid_x_train = tfvect.fit_transform(x_train)
    #tfid_x_test = tfvect.transform(x_test)
    input_data = [fnews]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred =  fake_news_det(message)
        print(pred)
        return render_template('index.html',prediction = pred)
    return None

if __name__ == '__main__':
    app.run(debug=True)