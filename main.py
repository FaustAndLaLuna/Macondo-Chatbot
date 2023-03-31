import embeddings_utils, json, os, pickle
import pandas as pd

from flask import Flask, request, current_app
from time import sleep
from dotenv import load_dotenv

load_dotenv()
openai_secret = os.environ['openai_secret']
openai_org = os.environ['openai_org']


app = Flask(__name__)

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

@app.route('/chatSilvestre', methods=['POST'])
def respond():
    if request.method == 'POST':
        data = request.json
        response = {'answer': embeddings_utils.get_ChatGPT_response(data['question'], data['history'])}
        embeddings = embeddings_utils.get_embedding(response['answer'])
        recommendations = []
        for _, row in df.iterrows():
            distances = embeddings_utils.distances_from_embeddings(embeddings, row['embeddings'])
            distances = sorted(distances)
            print(f'{distances[0]},{distances[1]}')
            if distances[0] < 0.12:
                recommendations.append({'handle':row['Handle'], 'title':row['Title'], 'img_src':row['Image_Src']})
        if len(recommendations) > 3:
            response['recommendations'] = recommendations[0:3]
        else:
            response['recommendations'] = recommendations
        print(recommendations)

    return json.dumps(response)

if __name__ == '__main__':
    df = pickle.load(open('products_final.pck', 'rb'))

    app.run(host='0.0.0.0', port=80)