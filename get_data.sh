#!/bin/sh

mkdir dataset  && \
cd dataset  && \
wget https://github.com/Emilien-mipt/FERplus-Pytorch/releases/download/0.0.2/fer_data.zip && \
unzip fer_data.zip  && \
rm fer_data.zip
