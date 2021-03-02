virtualenv -p $(which python3) 156_venv
. 156_venv/bin/activate
python -m pip install -r requirements.txt
python preprocess.py --trn 1203_firewall.csv 1210_firewall.csv 1216_firewall.csv --tst 156_SNACLab_5_0123_firewall.csv 156_SNACLab_5_0124_firewall.csv 156_SNACLab_5_0125_firewall.csv 156_SNACLab_5_0126_firewall.csv --output_trn train.csv --pretrained True
python main.py --trn train.csv --tst_src 156_SNACLab_5_0123_firewall.csv 156_SNACLab_5_0124_firewall.csv 156_SNACLab_5_0125_firewall.csv 156_SNACLab_5_0126_firewall.csv --eval False --pretrained True
deactivate
