import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for

from unzip import unzip

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            files = unzip(file)
            return render_template('uploaded.html', files=files)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
