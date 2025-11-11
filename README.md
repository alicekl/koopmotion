# KoopMotion

Code for **KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning** approach.  KoopMotion is a learning from demonstration work that enables learning smooth, stable motion policies (vector fields) from demonstration trajectories using Koopman operator theory. Evaluations are based on the **EPFL LASA handwriting dataset**.

---

## Installation

Clone this repository and install its requirements 
```
git clone https://github.com/alicekl/koop-motion.git
cd koop-motion
python -m pip install -r requirements.txt 
python -m pip install -e .
```
## Training a KoopMotion model
To train KoopMotion models for example shapes within `example_configuration_folder`, run the following.
```
python run_training.py example_configuration_folder
```
The above code trains a KoopMotion model, saves weights for future reconstruction under `trained_weights`, and under `figures` saves the constructed KoopMotion vector field, and trajectories from the vector field.

Note that we have included configuration files for all shapes of the LASA handwriting dataset. To train all shapes, replace `example_configuration_folder` in the above command with `configuration_files`, or for a sub-set, add/remove configuration folders into `example_configuration_folder`. 

We have only generated data for the example shapes in `example_configuration_folder`. See below for generating the rest of the dataset.

## Evaluating the model
To obtain the vector field from a set of trained weights, you may perform evaluation only:
```
python evaluation.py <path/to/configuration/file> <path/to/weights>
```

## Generating the sub-sampled (sparse) dataset
We have included training data for 2 examples. To generate all of the data for the EPFL LASA handwriting dataset and the sub-sampled data we used for our paper, and save the data to the `data` folder, run the following:
```
python generate_lasa_data.py
``` 

## References
This repository contains an implementation of the work below. If you find this repository useful, please cite the following work.

```
@inproceedings{li2025koopmotion,
  title={KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning},
  author={Li, Alice Kate and Silva, Thales C and Edwards, Victoria and Kumar, Vijay and Hsieh, M Ani},
  booktitle={Conference on Robot Learning},
  pages={2155--2169},
  year={2025},
  organization={PMLR}
}

```
