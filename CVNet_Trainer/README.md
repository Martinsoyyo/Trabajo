# Cement Trainer
Entrena la red de los archivos del dataset y almacena los pesos sinápticos en un archivos (CVNet_CementTrainer2).

## Links Readme.md Martín
- [https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a](https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a)
- [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)

## Argumentos
- --input={}
- --model={}
- --verbose (default=*OFF*)
- --epoch={}


### Input
El dataset es generado por otro programa que genera a partir de un lote clasificado de imágenes positivas y negativas. Los archivos por defecto son: `TRAIN_IMAGES.tensor`, `TRAIN_TARGET.tensor`,`TEST_IMAGES.tensor`, `TEST_TARGET.tensor`. Ver [CVNet_DatasetBuilder](ReconocimientoFisuras/src/Fisuras/CVNet_DatasetBuilder)

### Output
Al finalizar la ejecución se obtiene el archivo con los pesos sinápticos generados durante el entrenamiento. Por defecto en el archivo `model.pt`.

## Visual Studio - Windows 

## Linux
Ver los cambios implementados en el código en Visual Stuadio - Windows para que compile en Linux/Colab, trantando tanto errores como warnings en:
1. `changelog.txt` del 20200517 y 20200518
2. `logCompileBuild001.html`, salida de la primer compilación y build.


## Google Colab
1. Ver la preparación del entorno en [torch-install-GoogleColab.md](docs/torch/torch-install-GoogleColab.md)
2. Instalación de CMake 1.6
3. Cmdlineopt

```
# Libtorch 1.5 CUDA 10.0
!wget https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.5.0%2Bcu101.zip
!unzip libtorch-shared-with-deps-1.5.0+cu101.zip -d /content
```

```
# CMake version=3.16 build=5 
# Requerida para PCH

!apt remove --purge --auto-remove cmake
!version=3.16 
!build=5 
%cd ~/temp 
!wget https://cmake.org/files/v3.16/cmake-3.16.5.tar.gz 
!tar -xzvf cmake-3.16.5.tar.gz 
%cd cmake-3.16.5/
!./bootstrap 
!make -j$(nproc) 
!sudo make install
```

```
# Descarga librerías propias
%cd /content/
! git clone https://user:password@github.com/scativa/cmdlineopt/
```

### Google Drive a Colab
```
# Copia los dataset para entrenar y testear a /content
!mkdir /content/5y9wdsg2zt-1
!cp /content/drive/My\ Drive/Colab/5y9wdsg2zt-1/* /content/5y9wdsg2zt-1
```
```
# Copia el src a /content
!mkdir /content/CVNet_CementTrainer2
!mkdir /content/CVNet_CementTrainer2/build
!mkdir /content/common

Esto ya no funciona por el \\ en espacio de "*My Drive*", hay que subirlo de otra forma
~!cp /content/drive/My\ Drive/Colab/CVNet_CementTrainer2/* /content/CVNet_CementTrainer2~

```

### Build
```
# Compilación y build del código
%cd /content/CVNet_CementTrainer2/build
!cmake -DCMAKE_PREFIX_PATH=/content/libtorch -DCMDLINEOPT_PATH=/content/cmdlineopt ..
!cmake --build . --config Release
```
```
# Ejecutar el código
%cd /content/CVNet_CementTrainer2/build
!time ./CVNet_CementTrainer2 --input=/content/5y9wdsg2zt-1 --epoch=10
```
## Referencias
- Entorno Google Colab [torch-install-GoogleColab.md](/docs/torch/torch-install-GoogleColab.md)
- Ejemplo PyTorch en Visual Studio, Linux y Google Colab: [torch-test](/src/test/torch-test/)
### Links Readme.md Martín
- [https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a](https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a)
- [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)

