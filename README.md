# SIFT
Object detection using SIFT feature matching and then extraction using warp

## Getting Started  
  
Clone this repository and download all files

## One step solution
```
###Windows

Download Anaconda3-4.4.0-Windows-x86_64 and install it:

Create environment:
>conda env create -f py35_opencv_with_contrib_WIN.yml

to use environment:
>activate py35
>deactivate py35

###Linux 

Installing Anaconda
Link: https://www.digitalocean.com/community/tutorials/
how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

Create environment:
>conda env create -f py35_opencv_with_contrib_LINUX.yml

to use environment:
>source activate py35
>source deactivate py35

```
## Running the tests

```
python main.py <REF_path> <QRY_path> <name_of_query_image>

Ex 1: python main.py ../data/input.jpg ../data/query.jpg query

Result will be saved in the folder where is code
