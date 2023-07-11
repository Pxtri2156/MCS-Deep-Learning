import gdown
url = 'https://drive.google.com/uc?id=12tHvFWyu7eolNcFFGp9kywqS0tzTWzfq' 
out_path = 'dataset.zip'
gdown.download(url, out_path, quiet=False)