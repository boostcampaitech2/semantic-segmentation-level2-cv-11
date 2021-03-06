# <div align='center'>Data Preperation<div>
Aistages에서 제공하는 쓰레기 데이터를 사용합니다.

```bash
$ wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000078/data/data.zip
```

# <div align='center'>Quick Start Examples<div>

## <div align='left'> baseline code <div>
<details>
<summary> train </summary>

`configs/` 경로에 있는 `config.ini` 파일을 통해 hyper parameters 및 path 등을 수정합니다. 

`config.ini`를 바탕으로 train을 시작합니다.

```bash
$ python train.py --config_dir {config.ini path} 
```

or 

```bash
$ nohup python train.py --config_dir {config.ini path}&
```
</details>

<details>
<summary> inference </summary>

Train과 마찬가지로 `configs/` 경로에 있는 `config.ini` 파일을 통해 inference 를 수행합니다. 

```bash
$ python train.py --config_dir {config.ini path} --model_dir {model.pt path}
```
</details>



## <div align='left'> mmsegmentation <div>

<details>
<summary> train </summary>

- 경로: `/mmsegmentation`
1. 원하는 모델과 파라마티, 하이퍼 파라미터 config 세팅
2. `work_dir`을 지정, wandb project name, entity 설정
3. `config_dir` 지정후 코드 실행

```bash
$ python tools/train.py [config_dir]
```
</details>

<details>
<summary> inference </summary>

- 경로: `/mmsegmentation`
1. `--config_dir` Inference할 config 선택
2. `--epoch` Inference할 저장되어있는 pth파일 선택

```bash
$ python tools/inference.py --config_dir[config_dir] --epoch [epoch.pth_dir]
```
</details>



## <div align='left'> Tools <div>

<details>
<summary> SWA </summary>

SWA는 한 모델의 연속된 epoch or iteration 을 저장해 parameter의 가중치를 더한 후 평균 값을 도출 

`SWA/` 디렉토리에 SWA를 원하는 pth 파일 넣어서 `swa.py` 실행시 현 경로에 `swa.pth` 저장

```bash
$ python tools/swa.py
```
</details>

<details>

<summary> ensemble </summary>

- 경로 : `/mmsegmentation/tools`
    
    `--model_dir` : csv파일이 들어있는 경로
    
    `--save_dir` : 앙상블한 결과 저장 경로
    
    `--weight`  : 각 모델의 weight
    
```bash
$ python tools/ensemble.py --model_dir [model_dir : str] --save_dir [save_dir : str] --weight [weight : list]
```  
</details>

<details>
<summary> copy_paste  </summary>

- 경로 : `/mmsegmentation/tools`

```bash
$ python tools/ensemble.py 
```  
</details>

<details>
<summary> EDA </summary>

- 경로 : `/mmsegmentation/tools`
- `.ipynb` 파일
</details>

<details>
<summary> make_json </summary>

- 경로 : `/mmsegmentation/tools`

    `--original_json` : category를 추출할 json 파일 경로
    
    `--extract_json` : 추출된 데이터를 저장할 json 파일 경로
    
    `--category_num`  : 카테고리 인덱스
 
```bash
$ python tools/make_json.py --original_json {json file} --extract_json {extracted json file} --category_num {num of category}
```  
</details>
    
# <div align='center'>Reference<div>
    
- https://github.com/open-mmlab/mmsegmentation
    
- https://github.com/qubvel/segmentation_models.pytorch
