import os, openai, embeddings_utils, functools, pickle, glob, json
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from dotenv import load_dotenv
from typing import List
from nltk.corpus import stopwords
from time import sleep

load_dotenv()
openai_secret = os.environ['openai_secret']
openai_org = os.environ['openai_org']

stopwords = stopwords.words('english')

def get_stemmed(string:str) -> str:
    """Return a stemmed paragraphs."""
    return sent_tokenize(string)

def remove_stop_words(string: str) -> str:
    """Remove all stopwords in dict."""
    for word in stopwords:
        string = string.replace(f' {word} ', ' ')
    while('  ' in string):
        string = string.replace('  ', ' ')
    return string

def get_ChatGPT_datum(plant_name:str) -> str:
    """Gets ChatGPT percentage of how good the candidate fit is as a percentage."""
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            # {'role' : 'system', 'content' : ''},
            {'role' : 'user', 'content' : f'Describe las cualidades de la planta {plant_name} incluyendo su nombre cotidiano,'+
             ' tamaño, tipo de suelo, si tiene flor o no, color de flor, duración de su periodo de floreado, vida media, substrato, estética, en qué jardín quedaría bien, '+
             'si tiene frutos o no, color y textura, facilidad de cuidado y frecuencia de riego.'+
             ' También dime otras especies de planta con las que se vería bien.'}
            # {'role' : 'user', 'content' : ''}
        ],
        temperature=0.8
    )
    ans = res['choices'][0]['message']['content'].replace('\n', ' ')
    ans = ans.replace('  ', ' ')
    ans = ans.replace('  ', ' ')
    ans = ans.replace('  ', ' ')
    ans = ans.replace('  ', ' ')
    print(f'{plant_name} done!')
    return res['choices'][0]['message']['content']

def get_ChatGPT_Data(df:pd.DataFrame, MAX:int, MIN=0, column_name='description') -> None:
    try:
        temp_embeddings = []
        for i, row in df.iterrows():
            if i < MIN:
                continue
            if i > 0 and i % 50 == 0 and len(temp_embeddings) > 0:
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                temp_embeddings = []
            if i == MAX-1:
                temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_ChatGPT_plant_datum(row[column_name])})
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                break
            temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_ChatGPT_plant_datum(row[column_name])})
        dir_path = os.getcwd()
        files = glob.glob(os.path.join(dir_path, "*embeddings.pck"))
        for file in files:
            print(f"file={file}, len={len(pickle.load(open(file, 'rb')))}")

    except Exception as e:
        print(e)
        quit(500)

def break_paragraph_to_list(paragraph:str) -> List[str]:
    return sent_tokenize(paragraph)

def get_embeddings_from_generated_data(df:pd.DataFrame, MAX:int, MIN=0, column_name='description') -> None:
    try:
        temp_embeddings = []
        for i, row in df.iterrows():
            if i < MIN:
                continue
            if i > 0 and i % 50 == 0 and len(temp_embeddings) > 0:
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                temp_embeddings = []
            if i == MAX-1:
                temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_embeddings_from_list(break_paragraph_to_list(row[column_name]), engine='text-embedding-ada-002')})
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                break
            print(f'index: {i}')
            temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_embeddings_from_list(break_paragraph_to_list(row[column_name]), engine='text-embedding-ada-002')})
        dir_path = os.getcwd()
        files = glob.glob(os.path.join(dir_path, "*embeddings.pck"))
        for file in files:
            print(f"file={file}, len={len(pickle.load(open(file, 'rb')))}")
    except Exception as e:
        print(e)
        quit(500)

def get_embeddings(df:pd.DataFrame, MAX:int, MIN=0, column_name='description') -> None:
    try:
        temp_embeddings = []
        for i, row in df.iterrows():
            if i < MIN:
                continue
            if i > 0 and i % 50 == 0 and len(temp_embeddings) > 0:
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                temp_embeddings = []
            if i == MAX-1:
                temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_embedding(row[column_name], engine='text-embedding-ada-002')})
                print(f'Dumped up to i={i}')
                pickle.dump(temp_embeddings, open(f'{i}embeddings.pck', 'wb+'))
                break
            temp_embeddings.append({'index':i, 'embeddings':embeddings_utils.get_embedding(row[column_name], engine='text-embedding-ada-002')})
        dir_path = os.getcwd()
        files = glob.glob(os.path.join(dir_path, "*embeddings.pck"))
        for file in files:
            print(f"file={file}, len={len(pickle.load(open(file, 'rb')))}")
    except Exception as e:
        print(e)
        quit(500)

def get_embeddings_from_files(df:pd.DataFrame, column_name:str) -> pd.DataFrame:
    dir_path = os.getcwd()
    files = glob.glob(os.path.join(dir_path, "*embeddings.pck"))
    temp_embeddings = []
    for file in files:
        emb = pickle.load(open(file, 'rb'))
        print(f'{file}: {len(emb)}')
        temp_embeddings.extend(emb)

    temp_embeddings = sorted(temp_embeddings, key=lambda x:x['index'])

    emb_arr = []
    for embedding in temp_embeddings:
        emb_arr.append(embedding['embeddings'])
        print(f"{embedding['index']}{type(embedding['embeddings'])}")
    print(len(emb_arr))
    df[column_name] = emb_arr
    return df

def sort_and_clean_df(df:pd.DataFrame, column:str='description') -> pd.DataFrame:
    df = df[df[column].str.strip().astype(bool)]
    df.sort_values(by=column, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == '__main__':

    original_df = pd.read_excel('products_generated.xlsx')
    # original_df = sort_and_clean_df(original_df, column='Title')
    # original_df = sort_and_clean_df(original_df, column='Handle')

    # original_df['URL'] = original_df.apply(lambda row: f"https://macondo-shop.com/products/{row['Handle']}", axis=1)
    
    # # get_ChatGPT_Data(original_df, len(original_df.index), 100, 'Title')

    # original_df = get_embeddings_from_files(original_df, 'Generated_Data')

    # original_df.to_excel('products_generated.xlsx', index=False)

    # get_embeddings_from_generated_data(original_df, len(original_df.index), 0, 'Generated_Data')

    original_df = get_embeddings_from_files(original_df, 'embeddings')

    pickle.dump(original_df, open('products_final.pck', 'wb+'))

    # original_df.to_excel('products_final.xlsx')
    data = original_df.to_dict()
    # json.dump(data, open('products_final.json', 'w+'))

    print(original_df.head())
    
    # get_embeddings(original_df, len(original_df.index), 0)

    # print(len(original_df.index))

    # original_df = get_embeddings_from_files(original_df, 'embeddings')

    # for name, resume in resumes.items():

    #     df = original_df.copy()

    #     resume = remove_stop_words(resume)
    #     # resume_embeddings = embeddings_utils.get_embedding(resume)
    #     # pickle.dump(resume_embeddings, open(f'resume_embeddings_{name}.pck', 'wb+'))
    #     resume_embeddings = pickle.load(open(f'resume_embeddings_{name}.pck', 'rb'))
        


    #     df['distance'] = embeddings_utils.distances_from_embeddings(resume_embeddings, df.embeddings)
    #     df = df.append({'title_name':f'CV de {name}', 'country':'mx', 'location':'NaN','link':'NaN','embeddings':resume_embeddings,'distance':0}, ignore_index=True)
    #     df = df.sort_values(by='distance')

    #     matrix = np.array(df.embeddings.to_list())

    #     tsne = TSNE(n_components=2, perplexity=5, init='random')
        
    #     vis_dims = tsne.fit_transform(matrix)

    #     labels = np.array(df.title_name.to_list())

    #     df.iloc[0:17].to_excel(f'recomendaciones_{name}.xlsx', index=False)

    #     plot = embeddings_utils.chart_from_components(vis_dims, labels=labels, has_labels=True, strings=labels, has_strings=True)

    #     plot.add_trace(go.Scatter(x=[vis_dims[0][0]], y=[vis_dims[0][1]], mode = 'markers',
    #                         marker_symbol = 'star',
    #                         marker_size = 20))
        
    #     plot.add_trace(go.Scatter(x=[vis_dims[-1][0]], y=[vis_dims[-1][1]], mode = 'markers',
    #                         marker_symbol = 'circle',
    #                         marker_size = 15))

    #     plot.write_html(f'showcase_{name}.html')

    # # ans = recommendations_from_strings(openings)


