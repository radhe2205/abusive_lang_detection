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
All data preprocessed using script preprocess_all_data.py

```
python3 src/preprocess_all_data.py
```

To Train model checkout "train_nn.py" file.

NOTE: Run the train_nn.py file from the root directory.
```
python3 src/train_nn.py
```
