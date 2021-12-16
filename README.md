# abusive-language-detection
Detecting abusive language within tweets using the [OLID](https://sites.google.com/site/offensevalsharedtask/olid) and [SOLID](https://sites.google.com/site/offensevalsharedtask/solid) datasets.

- Rowan Lavelle | <rowan.lavelle@gmail.com>
- Radheshyam Verma | <radhe2205@gmail.com>

# Requirements

Make sure you are using python3.6 (highest version of python tensorflow works with). Create a virtual environment by running:

```
python3.6 -m venv proj_env
```

Once the environment is created run:

```
source proj_env/bin/activate
```

Then go ahead an install the requirements:

```
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

# Download Embeddings
```
sh download_embeddings.sh
```

# Running Code

## Preprocessing
All data preprocessed using script preprocess_all_data.py

```
python3 src/preprocess_all_data.py
```

## Transfer Learning
Within the transfer learning folder there are four files, files with `transfer` in them are the transfer models and the other files are the base model files.


### Multi-class transfer learning
To run `categorical_lstm.py` use either of the following
```
python3 categorical_lstm.py --mode=train
```

To train, test, and save the model to the save path in the dictionary
```
python3 categorical_lstm.py --mode=test
```
To test the saved model

This model is the multi-class base model.


To run `categorical_transfer_learning.py`, the multi-class transfer learning model run either of the following
```
python3 categorical_transfer_learning.py --mode=train --task=(a,b,c)
```
This will train, test, and save for the specified task, for example the task `a` flag would be `--task=a`. Must be lowercase

Testing works the same way
```
python3 categorical_transfer_learning.py --mode=test --task=(a,b,c)
```
Again the task must be lower case, this will report the results for the saved model.


### Language model transfer learning
To run `language_lstm_model.py` use the following flags
```
python3 categorical_lstm.py --mode=(test,train) --model-path=model.pth --temperature=float
```
the `train` flag will train, save and generate output from the model, `test` will load the trained model and output text.

the `--model-path` flag uses `model.pth` as the default so this does not need to be included, unless you want to train a new model. 

the `--temperature` flag should be a floating point value between `1e-10` to `1.0`, the default is `1.0`

To run the transfer learning model use the following flags
```
python3 language_transfer_learning.py --mode=train --task=(a,b,c)
```
This will train, test, and save for the specified task, for example the task `a` flag would be `--task=a`. Must be lowercase

Testing works the same way
```
python3 language_transfer_learning.py --mode=test --task=(a,b,c)
```
Again the task must be lower case, this will report the results for the saved model.


## Tri-Learning
To run the tri-learning file you only need to run the `tri_learner.py` file, this contains all the models. Use the following flags
```
python3 tri_learner.py --folder=saved_models/{folder} --mode=(train,test) --models(grid,diverse)
```

Here you should replace `{folder}` with the name of one of the tri-learning folders within `saved_models`.

If using the `train` flag you will need to specify either `grid` or `diverse`, these are the only models currently with built dictionaries, however you can reference the `key.txt` file in `saved_models` to change the dictionaries to train and test those models. By default the `test` settings just load in the saved results, since tri-learning takes a very long time. 


## Attention Model
To Train model checkout "train_nn.py" file.

NOTE: Run the train_nn.py file from the root directory.
```
python3 src/train_nn.py
```
