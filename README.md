# Semantic Segmentation of Faces

I have recently completed a data science course at General Assembly in London and for the past 5 weeks in the evenings I have been working on my final project. Cognisant of the rising tide of fake news / fake images and more recently deep fakes on the internet  I decided to create an image classifier that could spot whether pictures of faces had been photoshopped or not. 

This project ended up bifurcating into two parts with part 1 concerning pixel-based semantic segmentation of images of faces and part 2 developing the classifier to detect fake faces. Part 1 is contained within this repo and part 2 can be found [here](https://github.com/Rees451/faces-fake-vs-real).

A more detailed discussion of the methodology can be found in [this blog post](https://edward-rees.com/2019/12/12/segmentation.html)

## Project Intro/Objective

The purpose of this mini-project was to familiarise myself with working with images and aim to create a pixel-wise semantic segmentation classifier for images of faces. This project utilises the dataset found in [1] and [2].

### Methods Used

* Machine Learning (specifically random forest classifiers and K-Means clustering)

### Technologies

* Python
* jupyter
* Sklearn
* Scikit-image for image processing tasks

## Project Structure

1. Raw Data is kept [here](./data/raw) within this repo
2. Data processing/transformation scripts are being kept [here](./src)
3. Notebooks are kept [here](./notebooks)

## Featured Notebooks/Analysis/Deliverables

Analysis in this project was carried out in a jupyter notebook which documents each step. Please see the link below.

* [Main Notebook](./notebooks/face_segmentation.ipynb)
* [Blog Post](https://edward-rees.com/2019/12/12/segmentation.html)

## Contact

* If you want to contact me - reach out to me on [LinkedIn](www.linkedin.com/in/rees)

## References

[1] *Khalil Khan*, *Massimo Mauro*, *Riccardo Leonardi*,
**"Multi-class semantic segmentation of faces"**,
IEEE International Conference on Image Processing (ICIP), 2015
-- [**PDF**](https://github.com/massimomauro/FASSEG-repository/blob/master/papers/multiclass_face_segmentation_ICIP2015.pdf)

[2] *Khalil Khan*, *Massimo Mauro*, *Pierangelo Migliorati*, *Riccardo Leonardi*,
**"Head pose estimation through multiclass face segmentation"**,
IEEE International Conference on Multimedia and Expo (ICME), 2017
*In collaboration with [YonderLabs](http://www.yonderlabs.com)*
-- [**PDF**](https://github.com/massimomauro/FASSEG-repository/blob/master/papers/pose_estimation_by_segmentation_ICME2017.pdf)
