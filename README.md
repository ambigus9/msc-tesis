# msc-tesis
## Preparing schema
```
mkdir data/
```

## Getting data

- ucmercerd:
```
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF" -O ucmerced.zip && rm -rf /tmp/cookies.txt
unzip ucmerced.zip
rm -rv ucmerced.zip
```
- whu_rs19:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY" -O whu_rs19.zip && rm -rf /tmp/cookies.txt
unzip whu_rs19.zip
rm -rv whu_rs19.zip
```

- aid:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv" -O aid.zip && rm -rf /tmp/cookies.txt
unzip aid.zip
rm -rv aid.zip
```

- nwpu_resisc45
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd" -O nwpu_resisc45.zip && rm -rf /tmp/cookies.txt
unzip nwpu_resisc45.zip
rm -rv nwpu_resisc45.zip
```
