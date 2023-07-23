import pandas as pd
from sklearn.datasets import load_iris
from minio import Minio
from minio.versioningconfig import VersioningConfig, ENABLED

#
# dump data
#
iris = load_iris(as_frame=True)
X, y = iris["data"], iris["target"]
data = pd.concat([X, y], axis="columns")
data.sample(100).to_csv("iris.csv", index=None)

#
# minio client
#
url = "0.0.0.0:9000"
access_key = "minio"
secret_key = "miniostorage"
client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)

#
# upload data to minio
#
bucket_name = "raw-data"
object_name = "iris"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    config = client.set_bucket_versioning(bucket_name, VersioningConfig(ENABLED))

client.fput_object(bucket_name, object_name, "iris.csv")