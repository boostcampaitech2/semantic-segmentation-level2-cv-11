# <div align='center'>Quick Start Examples<div>

## <div align='center'> baseline code <div>
<details>
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

<details>
<summary> inference </summary>

Train과 마찬가지로 `configs/` 경로에 있는 `config.ini` 파일을 통해 inference 를 수행합니다. 

```bash
python train.py --config_dir {config.ini path} --model_dir {model.pt path}
```
</details>



## <div align='center'> mmsegmentation <div>

<details>
<summary> train </summary>

</details>

<details>
<summary> inference </summary>

</details>



## <div align='center'> Tools <div>

<details>
<summary> SWA </summary>

SWA는 한 모델의 연속된 epoch or iteration 을 저장해 parameter의 가중치를 더한 후 평균 값을 도출 

`SWA/` 디렉토리에 SWA를 원하는 pth 파일 넣어서 `swa.py` 실행시 현 경로에 `swa.pth` 저장

```bash
python tools/swa.py
```

</details>

<details>
<summary> ensemble </summary>

</details>