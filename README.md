# CatMatch

<!--INSERT PICTURE REPRESENTATIVE OF PROJECT-->
<!-- <div text-align="center">
<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.Khns8mi5ov-qN64yFABHmAHaE7%26pid%3DApi&f=1"></img>
</div> -->
<p text-align="center">
<a href="https://github.com/CogitoNTNU/README-template/blob/main/LICENSE" alt="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green"></img></a>

<a href="" alt="platform">
        <img src="https://img.shields.io/badge/platform-linux%7Cwindows%7CmacOS-lightgrey"></img></a>
<a href="" alt="version">
        <img src="https://img.shields.io/badge/version-0.0.1-blue"></img></a>
</p>

## About 
This project uses the power of recommender systems and convolutional neural networks to match you with your perfect cat.


### Prerequisites
- python 3.11 (suggest using [pyenv](https://github.com/pyenv-win/pyenv-win))
- [poetry](https://python-poetry.org/docs/) 1.6 

(Remember to set ` poetry config virtualenvs.in-project true`)

### Installation

```bash
poetry shell  # Creates a virtual environment
poetry install
```

The virtual environment (poetry shell) needs to be activated each time you work with the project.


### Installing pytorch
Pytorch needs to be installed manually, as the version depends on the the type of computer and if you have a graphics card with cuda. Go to [pytorch.org](https://pytorch.org/) and follow the instructions, install with pip inside the poetry environment.



### Plugins for development

This project uses the following tools for development which might require extensions in your editor.
- flake8
- black
- mypy

#### Running server
------
```bash
pip install . 
uvicorn catmatch.serve:app --reload # Runs the emojify/main.py file
```

#### Data
The data used for developing the movie recommender system is from [Kaggle](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset?select=ratings.csv). 

The dataset used for creating cat image embeddings is from [Kaggle](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection)


## Team
------
<!--INSERT PICTURE OF TEAM-->
<div text-align="center">
<img src="https://cogito-ntnu.no/static/img/projects/erpokerpfpwekwpkerwer.png"></img>
</div>

Right to left: [@example](https://github.com/Jonrodtang)    [@example](https://github.com/Jonrodtang)    [@example](https://github.com/Jonrodtang)    [@example](https://github.com/Jonrodtang)  
### Leader(s):
- [Ulrik Røsby](https://github.com/ulrik2204)

### Team members:
- [Marijan Soric](https://github.com/soricm)
- [Diogo Parcerias](https://github.com/pvdec)
- [Wilma Røise Huseby](https://github.com/Meeumi)
- [Adi Singh](https://github.com/adisinghstudent)
- [Erik Angus Usterud-Svendsen](https://github.com/erikangus)

## License
------
Distributed under the MIT License. See `LICENSE` for more information.

## Credits
------
Template by [@JonRodtang](https://github.com/Jonrodtang) for  [@CogitoNTNU](https://github.com/CogitoNTNU)  <p text-align="right">(<a href="#top">back to top</a>)</p>
