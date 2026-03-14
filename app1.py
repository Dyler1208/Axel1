from flask import Flask , render_template

app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello, World!'
@app.route('/n')
def Python():
    return "Hi"
@app.route('/h')
def index():
    return render_template
if __main__ == __name__:
    app.run()