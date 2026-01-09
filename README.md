# FUIMS Final

## Tópicos Usados

- /dji_m350/cameras/main/compressed
- /dji_m350/quaternion
- /dji_m350/velocity
- /dji_m350/gps

## Estruturas

### ENU

```c++
  struct ENU
{
    double x, y, z;
};
```

### Points

```c++
struct Points
{
    std::vector<int> ids;
    std::vector<cv::Point2f> pts;
};
```

### Keyframe

```c++
struct Keyframe
{
    int frameID = -1;
    ros::Time timestamp;
    cv::Mat R, t;
    Points points;
    cv::Mat greyImg;
};
```

### GpsSample

```c++
struct GpsSample
{
    ros::Time t; // bag time (robusto)
    double lat;  // radians
    double lon;  // radians
    double alt;  // meters
};
```

## Funções / Métodos

### undistortImage

Função responsável por remover a distorção da imagem recebida, com base nos parâmetros da câmara.  
Esta função guarda as imagens RGB e grayscale em variáveis globais e não retorna nada (tipo `void`).

**Input**: `sensor_msgs::CompressedImageConstPtr msg`  
**Output**: `none`

```c++
  void undistortImage(sensor_msgs::CompressedImageConstPtr msg)
    {
        cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        if (raw.empty())
        {
            currUndistortedGrey.release();
            currUndistortedRGB.release();
            return;
        }

        cv::undistort(raw, currUndistortedRGB, K, distCoeffs);
        cv::cvtColor(currUndistortedRGB, currUndistortedGrey, cv::COLOR_BGR2GRAY);
    }
```

### featureDetection

Função responsável por realizar a deteção de features utilizando o detetor ORB.  
**Passos**:

1. Deteção de pontos (até ao total definido aquando a inicialização do detetor).
2. Ordenação dos pontos detetados por ordem decrescente de qualidade
3. Definição de uma máscara para evitar pontos muito próximos.
4. Ciclo de seleção dos `ORB_N_BEST` pontos, verificando se pertencem à imagem e se não estão dentro de máscaras

Esta função vai guardar num vetor, declarado globalmente, os `ORB_N_BEST` pontos detetados e não retorna nada (tipo `void`).

**Input**: `sensor_msgs::CompressedImageConstPtr msg`  
**Output**: `none`

```c++
void featureDetection(const cv::Mat &img)
    {
        currPoints.pts.clear();
        currPoints.ids.clear();

        std::vector<cv::KeyPoint> kp;
        orb->detect(img, kp);

        std::vector<int> idx(kp.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kp[a].response > kp[b].response; });

        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

        int kept = 0;
        for (int id : idx)
        {
            cv::Point2f p = kp[id].pt;
            int x = cvRound(p.x);
            int y = cvRound(p.y);

            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
                continue;

            if (mask.at<uint8_t>(y, x) == 255)
            {
                currPoints.pts.push_back(p);
                currPoints.ids.push_back(nextFeatureID++);
                cv::circle(mask, cv::Point(x, y), radius, 0, -1);

                kept++;
                if (kept == ORB_N_BEST)
                    break;
            }
        }
    }
```

### replenishFeatures

Função que adiciona features novas ao vetor de pontos. Esta função é tipicamente chamada quando o número total de pontos está abaixo de um dado valor.  
Funcionamento semelhante ao 'featureDetection', detetando features na imagem recebida e adicionando as features necessárias ao vetor de pontos.

**Input**: `sensor_msgs::CompressedImageConstPtr msg`  
**Output**: `none`

### relativeMovementEstimation

**Passos**:

1. Verificações de condições para execuatar a função (Features suficientes e Matriz Essencial válida)
2. Chamada da função `cv::recoverPose` para obter a pose relativa entre dois keyframes
3. Processamento da matriz de rotação obtida para poder associa-la a uma matriz do tipo `gtsam::Rot3`
4. Processamento da translação obtida para poder associar a uma variável do tipo `gtsam::Point3`.
5. Determinação da escala, através da integração de velocidade entre os instantes de tempo `t0` e `t1`. Caso a escala seja muito baixa, a correção não é efetuada.
6. A translação é escala com base na escala obtida no passo anterior
7. Retorna uma varíavel do tipo `gtsam::Pose3`, com a Rotação e Translação.

**Input**: `std::vector<cv::Point2f> &prevValid,
                                            std::vector<cv::Point2f> &currValid,
                                            const ros::Time &t0,
                                            const ros::Time &t1`  
**Output**: `gtsam::Pose3`

### computeKFParallax
Função que calcula a paralaxe entre o keyframe atual e o frame atual. Serve para perceber a translação, em pixeis, entre as imagens.

**Passos**:
1. Verificação dos tamanho dos vetores presentes nos pontos do frame atual e do keyframe.
2. Criação de uma lookup table para os pontos do frame atual.
3. Execução de um loop que itera sobre os total de pontos existente no keyframe atual:
    - Guarda o ID do ponto atual e encontra um ponto com ID igual no frame atual
    - Caso o ponto exista, continua a iteração do loop atual, caso contrário, continua o loop.
    - Calcula a diferença em pixeis em xy e determina a distância euclidiana entre pontos.
    - Adiciona esta distância a uma varíavel de soma e incrementa um contador
4. Retorna a soma dividida pelo valor final do contador -> Valor da paralaxe média.

**Input**: `Keyframe &KF, Points &currPts`  
**Output**: `double parallax`


## Loop Principal
