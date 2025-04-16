# Indian Premier League 2008-2019


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)   [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/bprasad26/ipl_data_analysis/blob/master/LICENSE) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://www.lifewithdata.com/contact)

![IPL-WallPaper](https://github.com/bprasad26/ipl_data_analysis/blob/master/wallpaper.jpg?raw=true)




The Indian Premier League is a professional Twenty20 cricket league in India contested during March or April and May of every year by eight teams representing eight different cities in India. The league was founded by the Board of Control for Cricket in India in 2008.
&nbsp;
&nbsp;

## Project Website - [IPL Stats](https://bprasad26-ipl-stats.herokuapp.com/)
&nbsp;
&nbsp;






#### Installation 

Start by installing [Python](https://www.python.org/) and [Git](https://git-scm.com/downloads) if you have not already.


Next, clone this project by opening a terminal and typing the following commands (do not type the first $ signs on each line, they just indicate that these are terminal commands):

```sh
$ git clone https://github.com/bprasad26/ipl_data_analysis.git 
$ cd ipl_data_analysis
```


Next, create a virtual environment:


```sh
# create virtual environment
$ python3 -m venv ipl_venv
# Activate the virtual environment
$ source ipl_venv/bin/activate
```

Install the required packages
```sh
# install packages from requirements.txt file
$ pip install -r requirements.txt
```
Activate and Start jupyter Notebook
```sh
# activate venv for jupyter notebook
$ python -m ipykernel install --user --name=ipl_venv
# start jupyter notebook
$ jupyter notebook
# Note - at the top change the kernal to ipl_venv from the kernal dropdown if not done automatically.
```

deactivate the virtual environment, once done with your work
```sh
$ deactivate
```
to stay updated with this project
```sh
$ git pull
```

&nbsp;


import time
import numpy as np
from sentence_transformers import SentenceTransformer
from laser_encoders import LaserEncoder

# Test sentences in different languages
SENTENCES = [
    "This is a sentence.",           # English
    "Ceci est une phrase.",          # French
    "Dies ist ein Satz.",            # German
    "Esta es una oración.",          # Spanish
    "这是一个句子。"                   # Chinese
]

# List of transformer models to test
TRANSFORMER_MODELS = {
    "MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "BGE-M3": "BAAI/bge-m3"
}

def benchmark_transformer(name, model_name):
    print(f"\n--- {name} ---")
    model = SentenceTransformer(model_name)
    start = time.time()
    embeddings = model.encode(SENTENCES, convert_to_numpy=True)
    end = time.time()
    norm = np.linalg.norm(embeddings[0])
    return {
        "name": name,
        "dimensions": embeddings.shape[1],
        "time": round(end - start, 4),
        "norm": round(norm, 4),
        "langs": "100+",
        "type": "transformer"
    }

def benchmark_laser2():
    print(f"\n--- LASER2 ---")
    encoder = LaserEncoder()
    start = time.time()
    embeddings = encoder.encode_sentences(SENTENCES, lang='eng')
    end = time.time()
    norm = np.linalg.norm(embeddings[0])
    return {
        "name": "LASER2",
        "dimensions": embeddings.shape[1],
        "time": round(end - start, 4),
        "norm": round(norm, 4),
        "langs": "200+",
        "type": "laser2"
    }

def main():
    results = []

    # Benchmark transformer models
    for name, model_name in TRANSFORMER_MODELS.items():
        result = benchmark_transformer(name, model_name)
        results.append(result)

    # Benchmark LASER2
    results.append(benchmark_laser2())

    # Print summary
    print("\n--- Summary ---")
    print(f"{'Model':<12} {'Dims':<6} {'Time(s)':<8} {'Norm':<8} {'Langs':<8} {'Type'}")
    for r in results:
        print(f"{r['name']:<12} {r['dimensions']:<6} {r['time']:<8} {r['norm']:<8} {r['langs']:<8} {r['type']}")

if __name__ == "__main__":
    main()



