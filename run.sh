virtualenv -p $(which python3) 156_venv
. 156_venv/bin/activate
python -m pip install -r requirements.txt
python preprocess.py --trn /path/to/training_data/training_data.csv --tst /path/to/testing_data/testing_data.csv --output_trn train.csv --pretrained False
python main.py --trn train.csv --tst_src /path/to/testing_data/testing_data.csv --pretrained False
deactivate
