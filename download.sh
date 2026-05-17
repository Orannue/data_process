#!/bin/bash


EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` <filesToDownload> <chenlaneva@mails.cuc.edu.cn> <25B116A27F93D6036D46> [parallelDownloads=2]"
  exit $E_BADARGS
fi
filesToDownload=$1
username=$2
password=$3
parallelDownloads=$4

if [ $# -lt 4 ]
then
	parallelDownloads=2
fi

cat $filesToDownload | xargs -n 1 -P $parallelDownloads wget -crnH --cut-dirs=2 -q --user=$username --password=$password