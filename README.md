
# High-throughput(HTE) robotic titration workstation using computer vision


* An automated HTE robotic titration using a liquid-handling robot Opentrons(OT-2) and a standard webcam enables in-situ, affordable titration analyses. 

* Its modular design allows adaptability for materials chemsitry and integration into automated workflows, enhancing efficiency in chemical search.

Requirements and Installation
---------------------
### Clone the repo into your computer

```
git clone https://github.com/yuanlee0325/HTE-robotic-titration-using-computer-vision.git
```

### For working PC:
* Create a virtual environment

```
conda create -n ot2 python>=3.9
conda activate ot2
```

* Install requirements (mind your Python enviroment)

```
pip install -r requirements.txt
```

### For Opentrons(OT-2): 

* Transfer the required `.whl` files to OT-2

[pysmb-1.2.9.1-py3-none-any.whl](https://www.piwheels.org/simple/pysmb/pysmb-1.2.9.1-py3-none-any.whl#sha256=2a20a9d945efc2f6fe86afbe272f6dd4786344aca046b4ca2e98d519db817c20)
[tqdm-4.66.2-py3-none-any.whl](https://www.piwheels.org/simple/tqdm/tqdm-4.66.2-py3-none-any.whl#sha256=f78fd60412e4653a0be68fe9fc5424dc07ed3479bc765a3ab30e782d3d4dbd41)
[pyasn1-0.5.1-py2.py3-none-any.whl](https://www.piwheels.org/simple/pyasn1/pyasn1-0.5.1-py2.py3-none-any.whl#sha256=238ed5b4e0785e285c20dddcfd46ea3585d0ed25bd174d5737a08813db0de176)

```
ssh -i ot2_ssh_key root@ROBOT_IP
mkdir -p /var/data/user_storage/pysmb
scp -i ot2_ssh_key /local/path/to/package_name.whl/on/computer root@ROBOT_IP:/var/data/user_storage/pysmb
```

* Navigate to directory where you uploaded the package and install these packages 

```
ssh -i ot2_ssh_key root@ROBOT_IP
cd /path/to/packages
python3 -m pip install --target=/var/lib/jupyter/notebooks package_name.whl
```

* Upload files within `folder4ot2` to ot2
(1) Navigate to notebooks using http://ROBOT_IP:48888 or open Opentrons Software, click `device`, `Robot Settings`, `Advanced`, `Launch Jupyter notebook`subsequently
(2) Click `upload` to upload the essential files (config_files, h2o2, labware, utils, and H2O2_master_Pi_dev2.ipynb) saved in the `folder4ot2` 
(3) Replace the private robot/PC configuration within the `otsmb_aic.py` located in `/folder4ot2/utils`


Procedures 
-------------
![Schematic image](Schematic_graph.tif)


Documentation
-------------
The **video** of this HTE robotic chemical titration could be found in (https://doi.org/10.5281/zenodo.13825238)
Three different types of chemical titration were provided in the `data` folder, including:
(1) Redox Titration: H<sub>2</sub>O<sub>2</sub> determination titrated by KMnO<sub>4</sub>
(2) Acid-Base Titration: pH indicator using methylene orange
(3) Complexometric Titration: Ca<sup>2+</sup> titrated by EDTA


Citation
-------------
This project uses the following libraries and frameworks. 
- [NumPy](https://numpy.org)
- [Scipy](https://www.nature.com/articles/s41592-019-0686-2)
- [Pandas](https://pandas.pydata.org)
- [Matplotlib](https://matplotlib.org)
- [Scikit-Learn](https://scikit-learn.org)
- [PyTorch](https://pytorch.org)
- [OpenCV](https://opencv.org)
- [Pysmb](https://pysmb.readthedocs.io/en/latest/)
- [tqdm](https://doi.org/10.5281/zenodo.14231923)
- [pyasn1](https://pyasn1.readthedocs.io/en/latest/contents.html)


