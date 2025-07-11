ds=chess
mkdir $ds
wget https://storage.googleapis.com/$ds-tokenized/meta.pkl -P $ds/
wget https://storage.googleapis.com/$ds-tokenized/test.bin -P $ds/
wget https://storage.googleapis.com/$ds-tokenized/train.bin -P $ds/