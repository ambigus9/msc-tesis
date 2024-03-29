# msc-tesis
## Update enviroment
```
apt-get update
apt-get install wget zip
```

## Preparing schema
```
mkdir data/
```

## Getting data

- ucmercerd:
```
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF" -O data/ucmerced.zip && rm -rf /tmp/cookies.txt
unzip data/ucmerced.zip -d data/
rm -rv data/ucmerced.zip
```
- whu_rs19:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY" -O data/whu_rs19.zip && rm -rf /tmp/cookies.txt
unzip data/whu_rs19.zip -d data/
rm -rv data/whu_rs19.zip
```

- aid:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv" -O data/aid.zip && rm -rf /tmp/cookies.txt
unzip data/aid.zip -d data/
rm -rv data/aid.zip
```

- nwpu_resisc45
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd" -O data/nwpu_resisc45.zip && rm -rf /tmp/cookies.txt
unzip data/nwpu_resisc45.zip -d data/
rm -rv data/nwpu_resisc45.zip
```

## How to use

```
python baseline.py --yml config/nwpu.yml --gpu 5

docker exec -it miguel bash
cd /home/miguel/msc-tesis/ssl_satellital/exp_37/
python baseline.py --yml config/aid.yml --gpu 5

docker exec -it miguel bash
cd /home/miguel/msc-tesis/ssl_satellital/exp_37/
python baseline.py --yml config/whu_rs19.yml --gpu 5

docker exec -it miguel bash
cd /home/miguel/msc-tesis/ssl_satellital/exp_37/
python baseline.py --yml config/NWPU_RESISC45.yml --gpu 5
```
