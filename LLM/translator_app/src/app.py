import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
# Print model version.
print("MODEL - {}:{}".format(os.environ['PG_NAME'], os.environ['PG_VERSION']))

# Pipeline
from translate import translate
from trans_pipeline import TranslatePipeline

# Flask
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

# model load
translate_model = TranslatePipeline()
model,tokenizer = translate_model.models()


@app.route("/",  methods=['POST'])
@cross_origin(supports_credentials=True)
def main():

    # Get request.
    language = request.get_json()['language']
    text = request.get_json()['text']

    # translate.py input
    res_json = translate(language,text,model,tokenizer)

    return res_json

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
