# msc-tesis

## Method 1 - Create conda enviroment

```
conda create --name ssl-gleason python=3.7 -y
conda update -n base -c defaults conda -y
conda activate ssl-gleason
conda install jupyter -y
conda install ipykernel -y
pip install -r requirements.txt
python -m ipykernel install --user --name ssl-gleason
```

## Method 2 - Create docker instance

```
docker pull tensorflow/tensorflow:1.15.4-gpu-py3
docker run -it -v $(pwd):/home/ --name ssl-gleason-3 tensorflow/tensorflow:1.15.4-gpu-py3
cd home
```

**Note:** if GPUS are avaible and correctly configurated add `--gpus all` flag.

## Method 2 - Installing libraries
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Preparing schema
```
mkdir data/
```

## Getting data

- gleasson:

```
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LozwpWiRWUecarkQfoEAccpTW5VotRHm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LozwpWiRWUecarkQfoEAccpTW5VotRHm" -O gleasson.zip && rm -rf /tmp/cookies.txt
unzip gleasson.zip
rm -rv gleasson.zip
pwd
```

Note: Save this path on clipboard to use in next section.

## Setting master config

Modify exp_37/config/gleasson.yml file in following variables:

```
ruta_base=/home/my_user/ssl-gleasson/data
```

## How to use

```
cd ..
cd exp_37/
python baseline.py --yml config/gleasson.yml --gpu -1
```

## Docker Enviroment - How to use

```
docker start ssl-gleason-3
docker exec -it ssl-gleason-3 bash
cd /home/exp_37/
python baseline.py --yml config/gleasson.yml --gpu -1
```