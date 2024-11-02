sudo apt install wget curl unzip -y

curl -O http://images.cocodataset.org/zips/train2017.zip

curl -O http://images.cocodataset.org/zips/val2017.zip

curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip

echo "Downloaded COCO 2017 dataset"