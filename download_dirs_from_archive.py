import os
import boto3
import re
import string
import requests
from csv import reader
from urllib.parse import urlencode, quote_plus
from unidecode import unidecode

def get_matching_s3_objects(bucket, prefix='', suffix=''):
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix
    while True:
        resp = s3.list_objects_v2(**kwargs)
        try:
            contents = resp['Contents']
        except KeyError:
            return
        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield obj
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_matching_s3_keys(bucket, prefix='', suffix=''):
    for obj in get_matching_s3_objects(bucket, prefix, suffix):
        yield obj['Key']

def archive_download(bucket, prefix, asset, s3, metadata, debug=True):
    delete_table  = str.maketrans('', '', string.punctuation)
    basefile = asset['identifier']
    res = requests.get(asset['command'])
    if res.status_code == requests.codes.ok:
        content = res.content
        s3.Bucket(bucket).put_object(Key=prefix + basefile, Body=content)

session = boto3.session.Session()
s3 = session.resource('s3')
bucket = 'directory-ocr'
prefix = 'scans'
manifest_file = 'directories.txt'
manifest = {}
with open(manifest_file, encoding="utf-8") as f:
    lis=[line for line in reader(f)]        # create a list of lists
    for i,x in enumerate(lis[1:]):              #print the list items
        entry = dict(zip(lis[0], x))
        manifest[entry['identifier']] = entry

print("Manifest contains "+str(len(manifest))+" files")

s3_object_count = 0
notfound = []
for key in get_matching_s3_keys(bucket=bucket, prefix=prefix):#phenotypes', suffix='.tsv.bgz'):
    asset = os.path.basename(key)
    if(key != prefix):
        s3_object_count = s3_object_count + 1
        try:
            manifest.pop(asset, None)
        except ValueError as err:
            notfound.append(asset)

print(str(s3_object_count)+" files found in bucket")
print(str(len(manifest))+" files in manifest not found in bucket")

if(len(notfound) > 0):
    print(str(len(notfound))+" files found in bucket not in manifest")
    if(len(notfound) < 10):
        print(notfound)

if len(manifest) > 0:
    print("loading missing files")

for key, asset in manifest.items():
    archive_download(bucket, prefix, asset, s3, metadata, debug=False)
