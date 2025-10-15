# create the zip file that we will send to the bucket
zip -r models.zip ./models                          

# upload it to the bucket
gcloud storage buckets create gs://cant-model-assets --location=us-central1

#cleanup
rm models.zip