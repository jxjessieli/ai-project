# AI Big Project Run Instructions

Author:
* Ge Zichang
* Mirabel Tukiman
* Leng Sicong
* Li Jiaxi

### COVID-19 Retweet Prediction Challenge
- Competition page: https://competitions.codalab.org/competitions/25276#results

### Usage
---
If you want to train the model using the extracted features, run the following command in the project root dir.

```
python main.py --train
``` 

Note: the default pooling method is maxpooling. Please specify --meanpooling as an argument if you want to train using meanpooling. 

This will:
* Train one model and store the best-performing model to the root folder. 
* Evaluate the trained model on the test set based on the Mean Squared Log Error. 
* Save the predicted results on training set, validation set, and test set to csv files in the [./data/models/](https://github.com/jessiexiye/ai-project/tree/main/data/models) folder. 
* Save the losses-epoch figures in the [./figures/](https://github.com/jessiexiye/ai-project/tree/main/figures) folder. 

---
If you want to use the ensemble method based on the predicted results, run the following command in the project root dir. 

```
python ensemble.py --en_model regression
```

This will:
* Generate new predicted results using regression or average (to be specified in the command) based on two models' results stored in the csv files.
* Evaluate the new predicted results on the test set based on the Mean Squared Log Error.

---

If you want to evaluate the model using the trained weights, download the weight file from [best_model](https://drive.google.com/drive/folders/1G7mWNJYLoFPjFb_TXsqbJqefYjRK-uQD?usp=sharing) and put it under the root directory. Then run the following command in the project root dir. 

```
python main.py 
```

This will:
* Evaluate the trained model on the test set based on the Mean Squared Log Error.
* Save the losses-epoch figures in the [./figures/](https://github.com/jessiexiye/ai-project/tree/main/figures) folder. 

Note: This will evaluate the model starting with "False". To evaluate the model starting with "True", run the following:

```
python main.py --meanpooling
```

---
If you want to view the results using the user interface, run the following command in the ./gui folder. 

```
streamlit run gui_main.py
```