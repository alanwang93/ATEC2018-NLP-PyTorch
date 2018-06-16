# bash pack.sh <folder name>
rm -r submissions/$1
mkdir submissions/$1
echo $1> submissions/$1/model_name
cp *.py submissions/$1/
cp *.sh submissions/$1/
mkdir submissions/$1/log
cp log/$1 submissions/$1/log/$1
mkdir submissions/$1/checkpoints
cp checkpoints/$1.pkl submissions/$1/checkpoints/$1.pkl
cp checkpoints/$1.txt submissions/$1/checkpoints/$1.txt
mkdir submissions/$1/models
cp deep_models/*.py submissions/$1/models/
mkdir submissions/$1/data
mkdir submissions/$1/data/raw
cp data/raw/*.txt submissions/$1/data/raw/
cp data/*.py submissions/$1/data/
mkdir submissions/$1/data/processed
cp data/processed/vocab_char.pkl submissions/$1/data/processed/vocab_char.pkl
cp data/processed/vocab_word.pkl submissions/$1/data/processed/vocab_word.pkl
mkdir submissions/$1/data/extractors
cp data/extractors/*.py submissions/$1/data/extractors/
cd submissions/$1/
tar -czvf ../$1.tar.gz *
