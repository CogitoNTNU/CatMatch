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
Data in the data folder taken from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

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

## License
------
Distributed under the MIT License. See `LICENSE` for more information.

## Credits
------
Template by [@JonRodtang](https://github.com/Jonrodtang) for  [@CogitoNTNU](https://github.com/CogitoNTNU)  <p text-align="right">(<a href="#top">back to top</a>)</p>
