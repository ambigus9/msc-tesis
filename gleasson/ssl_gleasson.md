# msc-tesis

## Installing libraries
```
su
pip install --upgrade pip
pip install -r requirements.txt
exit
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
```

## How to use

```
docker start miguel-tf2
docker exec -it miguel-tf2 bash
su

cd /home/miguel/msc-tesis/gleasson/
python baseline.py --yml config/gleasson.yml --gpu 5
```