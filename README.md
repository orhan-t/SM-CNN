## SM-CNN: Hyperspectral Image Denoising via Self-Modulating Convolutional Neural Networks


### Installation
* Install dependencies:
```bash
pip install -r requirements.txt
```
### Train and Test
* Put your data into the 'datasets' directory 
* Creat train, val and test patches ((check get_patch_wdc.py and get_patch_wdc_test.py))
* Open the model directory in terminal
* Set python environment
* Train example:
```bash
python .\train.py --config-file .\gauss_blind_config.yaml
```
* Test example: 
```bash
python .\predict.py --config-file .\gauss_blind_config.yaml
```