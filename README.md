# Duplicated Space Server   
VR 사용자가 사용할 클라이언트인 Duplicated Space Client의 서버 역할부   
Firebase와의 연동을 위해 자신의 Firebase Storage와 Firebase Realtime Database 정보를 입력해 사용해야 함  
   

## Setup
### 0. (Optional) Conda Setup
자신의 환경에 충돌이 없도록 하기 위해서, 콘다 환경을 생성 후 작업을 권함.   

### 1. Dependence Setup
- [GaussianSpaltting](https://github.com/graphdeco-inria/gaussian-splatting)   
- [InstantSplat](https://github.com/NVlabs/InstantSplat)   
- [SAM2](https://github.com/facebookresearch/sam2)
### 2. Server Setup
- Firebase
```bash
$ pip install firebase-admin   
```
### 3. Firebase Setup
[Firebase](https://firebase.google.com/?gad_source=1&gclid=CjwKCAiAg8S7BhATEiwAO2-R6vWLGnjbbHKCBJaHCs3i6o6FEOZ6I53XdFqubCrGM-2KvDIRQKLh4RoC74gQAvD_BwE&gclsrc=aw.ds)에서 자신의 프로젝트 생성 후 google-service.json을 다운. 이후의 아래의 작업 수행   
Server/FirebaseServerSpace.py와 Server/FirebaseServerObject.py, segment-anything-2/capstone/capstone.py의
```bash
cred = credentials.Certificate('***')
firebase_admin.initialize_app(cred, {
    'databaseURL': '***',
    'storageBucket': '***'
})
```
에서의 '***'에 자신의 데이터 입력   
### 4. add alias
- 콘다 환경에서 세팅한 경우
```bash
$ echo "alias capstone_space='conda activate {conda env name} && cd ~/Projects/Capstone/Firebase_Ubuntu/ && python FirebaseServerSpace.py'" >> ~/.bashrc
```
```bash
$ echo "alias capstone_object='conda activate {conda env name} && cd ~/Projects/Capstone/Firebase_Ubuntu/ && python FirebaseServerObject.py'" >> ~/.bashrc
```
- 베이스 환경에서 세팅한 경우
```bash
$ echo "alias capstone_space='cd ~/Projects/Capstone/Firebase_Ubuntu/ && python FirebaseServerSpace.py'" >> ~/.bashrc
```
```bash
$ echo "alias capstone_object='cd ~/Projects/Capstone/Firebase_Ubuntu/ && python FirebaseServerObject.py'" >> ~/.bashrc
```


## Usage
1. 공간 생성용   
```bash
$ capstone_space
```
2. 물체 생성용
```bash
$ capstone_object
```


## Example Video
https://youtu.be/9ZbRM-3eRyU
