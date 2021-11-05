# <div align='center'>Quick Start Examples<div>

### baseline code
<details open>
<summary> train </summary>

`configs/` 경로에 있는 `config.ini` 파일을 통해 hyper parameters 및 path 등을 수정합니다. 

`config.ini`를 바탕으로 train을 시작합니다.

```bash
python train.py --config_dir {config.ini path} 
```

or 

```bash
nohup python train.py --config_dir {config.ini path}&
```
</details>

<details open>
<summary> inference </summary>

Train과 마찬가지로 `configs/` 경로에 있는 `config.ini` 파일을 통해 inference 를 수행합니다. 

```bash
python train.py --config_dir {config.ini path} --model_dir {model.pt path}
```
</details>



### mmsegmentation 

<details open>
<summary> train </summary>

</details>

<details open>
<summary> inference </summary>

</details>



### Tools 

<details open>
<summary> SWA </summary>

`SWA/` 디렉토리에 SWA를 원하는 가중치 

</details>

<details open>
<summary> ensemble </summary>

</details>