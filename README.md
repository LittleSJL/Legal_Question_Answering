# Folder structure
There are five folders:
- `data`: 
	- `data_token.json`: Handbook QA with token-level answer annotation for QG questions
	- `data_sentence.json`: Handbook QA with sentence-level answer annotation for QG questions
- `model`: 
	- `reader`: include the raw bert model (2_128) used for training and evaluating
	- `retriever`: include the retirever model (tf-idf matrix)
- `output`: folder to store output results.
- `pipeline`: include the main coding files for pipeline (retriever and reader)
- `scripts`: include coding files to train/evaluate the retriever/reader

# Simple instructions on how to use the code
- For retriever:
	- run `python scripts/retriever/build_text.py` file to prepare the text
	- run `python scripts/retriever/build_db.py` file to build the db
	- run `python scripts/retriever/build_tfidf.py` file to build the retriever model (tf-idf)
	- run `python scripts/retriever/demo.py` file to get the evaluation results
- For reader:
	- run `python scripts/reader/prepare_file.py` to prepare the files for training and evaluating
	- run `python scripts/reader/demo.py --train` to train the model (with different settings)
	- run `python scripts/reader/demo.py --predict --model_dir output/reader\model.ckpt-77` to use the model to predict (with different settings)
	    - `model.ckpt-77` is the model you have just saved (you can use any existing models)
	- run `python scripts/reader/evaluate.py --file output/reader/prediction_pipeline.json --refor Original` to get the evaluation results
	    - `prediction_pipeline.json` is the prediction file you have just saved (you can use any existing prediction files)





















