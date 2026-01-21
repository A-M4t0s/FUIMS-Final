# FUIMS Final

## Execução
Para executar o algoritmo de VIO executar os seguintes comandos:

```bash
catkin build fuims # Compilar o package
roslaunch fuims real_time_vio.launch
```

Para facilitar a alteração dos estados, executar num terminal:

```bash
rosrun fuims vio_state_publisher
```

## Resultados
No final de cada ciclo, é gerado um ficheiro CSV numa pasta indicada dentro do código escrito em 'fuims_vio_v6.cpp'. Para alterar o path, basta alterar o código e recompilar.

Para a visualização destes resultados, basta executar num terminal:
```bash
python3 catkin_ws/src/fuims/scripts/evaluate_vio_results.py
```

Este comando exibe um CLI que permite dar load aos ficheiros .csv presentes na pasta indicada no ficheiro Python correspondente (alterar em concordância com a localização da pasta pretendida).  
É possível gerar um outro ficheiro CSV na CLI que contem uma seleção de resultados, para posterior comparação na GUI exibida por:
```bash
python3 catkin_ws/src/fuims/scripts/compare_vio_results.py
```
