from flask import Flask, render_template
import pickle as pkl

app = Flask(__name__)

with open ('mm1.pkl','rb') as file:
    data=pkl.load(file)

@app.route('/')
def iris():
    return render_template('iris.html')
@app.route('/pred')
def iris1():
    petal_length = request.form['petal_length']
    sepal_length = request.form['sepal_length']
    petal_width = request.form['petal_width']
    sepal_width = request.form['sepal_width']
    model_choice = request.form['model_choice']

    # Clean the data by convert from unicode to float
    sample_data = [sepal_length,sepal_width,petal_length,petal_width]
    clean_data = [float(i) for i in sample_data]

    # Reshape the Data as a Sample not Individual Features
    ex1 = np.array(clean_data).reshape(1,-1)
    prediction = predict(ex1)

    return render_template('iris.html', predict_text = ' flower species is {}'.format(prediction))
     

if __name__ == "__main__":
    app.run()