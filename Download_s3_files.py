# To run in ec2 instance to download ALL files from s3 bucket
import boto3
import os
# new directories
directories = ['../originals', '../transforms', '../motion']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
)
s3 = session.resource('s3')
# s3.download_file('nuradmris', 'OBJECT_NAME', 'FILE_NAME')
my_bucket = s3.Bucket('nuradmris')

# my_bucket.download_file('originals/M01.nii', '../originals/M01.nii')
for s3_object in my_bucket.objects.all():
    # Need to split s3_object.key into path and file name, else it will give error file not found.
    key = s3_object.key
    trans_limit = 5 # limit to number of transformed mris downloaded
    trans_cnt = 0
    if not key.endswith('/'):
        if key.startswith('transforms/') and trans_cnt < trans_limit:
            print('Uploading: ' + key)
            my_bucket.download_file(key, '../' + key)
            trans_cnt += trans_cnt
        elif not key.startswith('transforms/'):
            print('Uploading: ' + key)
            my_bucket.download_file(key, '../' + key)
