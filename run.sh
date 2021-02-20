virtualenv -p $(which python3) 156_venv
. 156_venv/bin/activate
python -m pip install -r requirements.txt
#python preprocess.py --trn 1203_firewall.csv 1210_firewall.csv 1216_firewall.csv --tst 156_SNACLab_1_0123_firewall.csv 156_SNACLab_1_0124_firewall.csv 156_SNACLab_1_0125_firewall.csv 156_SNACLab_1_0126_firewall.csv --output_trn train.csv --pretrained True
#python main.py --trn train.csv --tst_src 156_SNACLab_1_0123_firewall.csv 156_SNACLab_1_0124_firewall.csv 156_SNACLab_1_0125_firewall.csv 156_SNACLab_1_0126_firewall.csv --eval False --pretrained True

python preprocess.py --trn ../nad/train_no.csv --tst ../nad/test_no.csv --output_trn train.csv
python main.py --trn train.csv --tst_src ../nad/test_no.csv

deactivate
