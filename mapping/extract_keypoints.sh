#!/bin/bash

echo "$@"

if [ "$#" -ne 2 ]
then
	echo "Arguments invalid"
	exit
fi

r2d2_path="$1"
data_dir="$2"

n=${#r2d2_path}

if [ "${r2d2_path: -1 : 1}" == "/" ]
then
	r2d2_path="${r2d2_path: 0 : ${n}-1}"
fi

if [ "${r2d2_path: -3 : 3}" == ".py" ]
then
	echo "invalid path ${r2d2_path}"
	exit
fi

script_path=$r2d2_path"/extract.py"
model_path=$r2d2_path"/models/r2d2_WASF_N16.pt"
n=${#data_dir}

if [ "${data_dir: -1 : 1}" == "/" ]
then
	data_dir="${data_dir: 0 : ${n}-1}"
fi

original_image_path=$data_dir"/cam_out/original_images"

if [ ! -d $original_image_path ]
then
	echo "directory : "$original_image_path" not found"
	exit	
fi

images="image_list.txt"
rm -rf $images
touch $images

for image in $original_image_path/*; do
	echo $image >> $images
done

python3 $script_path --model $model_path --min-size 0 --max-size 9999 --min-scale 0.3 --max-scale 1.0 --images $images --top-k 3000

descriptors=$data_dir"/cam_out/descriptors"
keypoints=$data_dir"/cam_out/keypoints"
scores=$data_dir"/cam_out/scores"
temp=$data_dir"/cam_out/temp"

rm -rf $descriptors && rm -rf $keypoints && rm -rf $scores && rm -rf $temp
mkdir $descriptors && mkdir $keypoints && mkdir $scores && mkdir $temp

for zips in $original_image_path/*.r2d2; do
	unzip -q $zips -d $temp
	x=${zips##*/}
	file_name=${x%.*.r2d2}
	mv $temp"/descriptors.npy" $descriptors"/"$file_name".npy"
	mv $temp"/keypoints.npy" $keypoints"/"$file_name".npy"
	mv $temp"/scores.npy" $scores"/"$file_name".npy"
	rm $temp"/imsize.npy"
	rm $zips
done

rm -rf $temp
rm $images
