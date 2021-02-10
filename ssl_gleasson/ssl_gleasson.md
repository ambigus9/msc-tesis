# msc-tesis
## Preparing schema
```
mkdir data/
```

## Getting data

- gleasson:
```
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF" -O ucmerced.zip && rm -rf /tmp/cookies.txt
unzip ucmerced.zip
rm -rv ucmerced.zip
```

## How to use

```
docker exec -it miguel bash
cd /home/miguel/msc-tesis/ssl_satellital/exp_37/
python baseline.py --yml config/gleasson.yml --gpu 5
```