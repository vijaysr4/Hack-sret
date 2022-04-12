from flask import  Flask, render_template, request
import pandas as pd
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run()