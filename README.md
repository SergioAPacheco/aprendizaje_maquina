# Predicción de graduación de beneficiarios de becas — Antioquia

Aplicación web que estima si un beneficiario de un apoyo económico a la educación
superior en Antioquia culminará su programa académico.

Proyecto Integrador de Minería de Datos · Maestría en Ciencia de Datos · UPB 2026
Metodología CRISP-DM.

**Andrés Camilo Jiménez Guerra · Cynthia Gaviria Castaño · Sergio Andrés Pacheco Márquez**

---

## El problema

La Corporación Gilberto Echeverri Mejía administra subsidios, becas condicionadas y
créditos condonables para población de Antioquia que reside fuera de Medellín. En el
conjunto de datos disponible, cerca del 51 % de los beneficiarios no figura como
graduado del programa para el que recibió el apoyo.

El objetivo es identificar tempranamente a quienes tienen mayor riesgo de no culminar,
para activar acompañamiento antes de que el apoyo se pierda.

## Los datos

[Beneficiarios de becas y créditos de programas de acceso a la educación superior de
Antioquia](https://www.datos.gov.co/d/ya7f-466y) — Datos Abiertos Colombia.

La versión usada en este trabajo se descargó el 8 de marzo de 2026: 14.566 registros y
17 variables. Tras eliminar 14 duplicados y construir cuatro atributos derivados
—`edad_beneficiario`, `movilidad_territorial` y dos indicadores de Área
Metropolitana— el conjunto de modelado queda en **14.552 registros y 15 variables**,
equivalentes a 247 columnas después de la codificación.

## El modelo

Se compararon once clasificadores bajo un mismo protocolo: partición estratificada
70/30 con semilla fija, búsqueda de hiperparámetros por `GridSearchCV` con validación
cruzada estratificada de 5 particiones sobre el conjunto de entrenamiento, y una
única evaluación final sobre el conjunto de prueba.

Métrica de selección: **AUC-ROC**, con F1 macro como desempate.

| Modelo | AUC-ROC (test) | F1 macro | Accuracy |
|---|---|---|---|
| **Votación (soft)** | **0,7954** | **0,7205** | **0,7206** |
| XGBoost | 0,7929 | 0,7092 | 0,7098 |
| Bagging con KNN | 0,7884 | 0,7131 | 0,7135 |
| Red Neuronal (MLP) | 0,7858 | 0,7048 | 0,7050 |
| Random Forest | 0,7833 | 0,7019 | 0,7022 |
| SVM (poly, degree=3) | 0,7819 | 0,7212 | 0,7215 |
| Regresión Logística | 0,7719 | 0,6978 | 0,6981 |
| KNN | 0,7700 | 0,6999 | 0,7002 |
| AdaBoost | 0,7584 | 0,6875 | 0,6876 |
| Árbol de Decisión | 0,7410 | 0,6605 | 0,6642 |
| Baseline (Dummy) | 0,5000 | 0,3373 | 0,5089 |

El modelo desplegado es una **votación blanda** sobre cinco familias distintas:
regresión logística, SVM polinómico, red neuronal, Random Forest y XGBoost, cada uno
con los hiperparámetros hallados por `GridSearchCV`. Promedia sus probabilidades.

Gana en las dos métricas declaradas: AUC-ROC (0,7954 contra 0,7929 de XGBoost) y
F1 macro como desempate (0,7205 contra 0,7092). La coherencia entre validación
cruzada (0,7979) y prueba (0,7954) indica que generaliza bien.

Bagging y boosting combinan muchas copias de un mismo aprendiz; la votación combina
familias que se equivocan en registros distintos, y por eso supera a los cuatro
ensambles homogéneos evaluados.

## Cómo correr la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Las versiones de `scikit-learn`, `xgboost` y `numpy` están fijadas con `==` porque son
las que serializaron `modelo-clasificacion.pkl`. Con otras versiones el modelo falla al
cargar: el artefacto es un `VotingClassifier`, y su reconstrucción depende del formato
interno de cada librería.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación Streamlit. Captura los datos en dos bloques y devuelve la predicción. |
| `modelo-clasificacion.pkl` | Modelo (29 MB: cinco estimadores), `LabelEncoder`, nombres de las 247 columnas en orden de entrenamiento y `MinMaxScaler` de la edad. |
| `requirements.txt` | Dependencias con versiones fijadas. |
| El CSV | El dataset original tal como se descargó. |

El `.pkl` guarda una lista de cuatro elementos y ese orden es el contrato con la app:

```python
modelo, label_encoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))
```

`variables` es lo que alinea las columnas del despliegue con las del entrenamiento:
la app arma su dataframe con `get_dummies(drop_first=False)` y luego hace
`reindex(columns=variables, fill_value=0)`.

## Limitaciones

Conviene leerlas antes de usar la salida del modelo para cualquier decisión.

**El modelo se equivoca en el 27,9 % de los casos.** La exactitud es del 72,1 %.

**La variable objetivo está censurada en el tiempo.** La tasa de graduación pasa de
67,9 % en la convocatoria de 2013 a 0,0 % en las de 2023 y 2024. Quien recibió el
apoyo hace poco no ha tenido tiempo material de graduarse, de modo que `GRADUADO = NO`
mezcla a quien abandonó con quien todavía está estudiando. Parte de lo que el modelo
aprende como riesgo de deserción es, en realidad, antigüedad de la cohorte.

**Hay variables sensibles entre los predictores.** Género, estrato, grupo étnico y
condición de víctima del conflicto entran al modelo, y las tasas de graduación difieren
de forma marcada entre grupos.

**No hay grupo de comparación.** El conjunto contiene únicamente a quienes recibieron
el apoyo, así que el modelo no permite estimar el efecto del auxilio, solo predecir el
desenlace dentro de esa población.

Por todo lo anterior, esta herramienta está pensada para **priorizar acompañamiento y
seguimiento**, no para negar, condicionar o retirar un beneficio.
