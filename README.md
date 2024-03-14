# Documentação do Modelo MFA_RNN

Este repositório contém o código para o modelo MFA_RNN usado para a previsão da próxima categoria de ponto de interesse (POI) no contexto do projeto de detecção de POI. Este documento visa esclarecer o uso de 19 inputs no modelo MFA_RNN, bem como a discrepância encontrada em relação ao número de inputs utilizados em diferentes partes do código.

## Modelo MFA_RNN

O arquivo [mfa_rnn.py](https://github.com/claudiocapanema/poi_detection/blob/next_place_prediction/model/next_poi_category_prediction_models/gowalla/poi_rgnn/mfa_rnn.py) implementa o modelo MFA_RNN. O modelo utiliza 19 inputs para prever a próxima categoria de POI com base em diferentes características.

### Inputs Utilizados no Modelo MFA_RNN

Os 19 inputs utilizados no modelo são:

1. Input 1
2. Input 2
3. ...
4. Input 19

Cada input representa uma característica específica que é considerada na previsão da próxima categoria de POI.

## Discrepância no Número de Inputs

A discrepância no número de inputs é observada na implementação do método `train_test_split` no arquivo [next_poi_category_prediction_domain.py](https://github.com/claudiocapanema/poi_detection/blob/next_place_prediction/domain/next_poi_category_prediction_domain.py). Neste método, apenas 13 inputs são exemplificados, enquanto o modelo MFA_RNN utiliza 19 inputs.

### Exemplo de Discrepância

```python
# Trecho do código em next_poi_category_prediction_domain.py

# Inputs para train_test_split
inputs = [
    'input1',
    'input2',
    ...,
    'input13'
]
```
Houve a tentativa de remoção dos inputs não úteis ao modelo, mas isso muda completamente a camada na rede neural. Qual era a principal ideia da camada da rede utilizando os 13 inputs ?


