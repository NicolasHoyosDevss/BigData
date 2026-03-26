# SDSS Machine Learning Pipeline

Proyecto de Machine Learning modular y reproducible construido sobre el dataset astronomico `sdss_sample.csv`.

El objetivo del proyecto es implementar un pipeline completo que permita:

- cargar y limpiar datos
- entrenar varios modelos
- evaluar sus resultados
- guardar metricas y visualizaciones
- dejar la base lista para reproducibilidad y automatizacion en fases posteriores

## Estado Actual

Hasta este punto del proyecto estan cubiertas las siguientes fases:

- Fase 1: estructura base, carga del dataset, inspeccion y preprocesamiento inicial
- Fase 2: clasificacion con KNN (`k=5`)
- Fase 3: regresion lineal
- Fase 4: clustering con KMeans (`k=3`)
- Fase 5: guardado de metricas y graficas en `outputs/`
- Fase 6: integracion end-to-end en `main.py`
- Fase 7: Dockerizacion basica del proyecto
- Fase 8: pipeline basico con Jenkins

Pendiente:

- Ajustes finales segun entorno de despliegue

## Estructura Del Proyecto

```text
Parcial/
|-- main.py
|-- README.md
|-- sdss_sample.csv
|-- outputs/
|   |-- metrics/
|   `-- plots/
`-- src/
    |-- __init__.py
    |-- preprocessing.py
    |-- classification.py
    |-- regression.py
    |-- clustering.py
    `-- reporting.py
```

## Dataset

El dataset utilizado es `sdss_sample.csv`, con observaciones astronomicas del Sloan Digital Sky Survey.

Columnas principales:

- `u`, `g`, `r`, `i`, `z`: magnitudes fotometricas
- `redshift`: corrimiento al rojo
- `class`: clase astronomica real (`Galaxy`, `Star`, `QSO`)
- `snr_r`: relacion senal/ruido en banda `r`
- `extinction_r`: extincion en banda `r`

## Flujo Del Pipeline

El flujo esta centralizado en [`main.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/main.py).

Orden de ejecucion:

1. Cargar el dataset
2. Inspeccionar los datos originales
3. Limpiar y preparar el dataset
4. Ejecutar clasificacion con KNN
5. Ejecutar regresion lineal
6. Ejecutar clustering con KMeans
7. Construir un reporte consolidado en memoria
8. Guardar metricas y graficas en `outputs/`

Importante:

- los modelos no guardan archivos directamente
- cada modulo devuelve resultados al `report`
- el modulo [`src/reporting.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/reporting.py) toma ese `report` y genera los archivos finales

## Modulos

### [`src/preprocessing.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/preprocessing.py)

Contiene la logica inicial de datos:

- `load_data()`
- `inspect_data()`
- `clean_data()`

Responsabilidades:

- cargar el CSV con `pandas`
- validar columnas obligatorias
- revisar tamano, tipos, nulos y duplicados
- convertir columnas numericas
- dejar el dataset listo para las siguientes etapas

### [`src/classification.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/classification.py)

Implementa clasificacion supervisada con:

- `KNeighborsClassifier`
- `k = 5`
- escalado con `StandardScaler`

Metricas:

- `accuracy`
- `confusion_matrix`

### [`src/regression.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/regression.py)

Implementa regresion supervisada con:

- `LinearRegression`
- escalado con `StandardScaler`

Objetivo de regresion:

- `redshift`

Metricas:

- `MSE`
- `R2`

Nota:

- este modelo se usa como baseline
- puede generar predicciones negativas, porque la regresion lineal no restringe la salida a valores no negativos

### [`src/clustering.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/clustering.py)

Implementa clustering no supervisado con:

- `KMeans`
- `k = 3`
- escalado con `StandardScaler`

Evaluacion y apoyo visual:

- `silhouette_score`
- tamano de clusters
- comparacion `cluster_vs_class`
- proyeccion 2D con PCA para graficar

### [`src/reporting.py`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/src/reporting.py)

Se encarga de persistir resultados en disco.

Responsabilidades:

- guardar reportes en JSON
- guardar resumen en TXT
- generar las graficas en PNG
- separar metricas compactas de datos auxiliares para visualizacion

## Outputs Generados

Despues de ejecutar el pipeline, el proyecto crea archivos dentro de `outputs/`.

### `outputs/metrics/`

- `pipeline_report.json`: reporte consolidado y compacto del pipeline
- `summary.txt`: resumen corto de resultados
- `classification_metrics.json`: metricas de clasificacion
- `regression_metrics.json`: metricas de regresion
- `regression_plot_data.json`: datos usados para la grafica de regresion
- `clustering_metrics.json`: metricas de clustering
- `clustering_plot_data.json`: datos usados para graficas de clustering

### `outputs/plots/`

- `classification_confusion_matrix.png`
- `regression_actual_vs_predicted.png`
- `clustering_projection.png`
- `clustering_vs_class.png`

## Como Ejecutarlo

Desde PowerShell, dentro de la carpeta del proyecto:

```powershell
& 'C:\Users\nicoh\AppData\Local\Python\bin\python.exe' -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib numpy
python main.py
```

## Ejecucion Parcial

Tambien se pueden saltar etapas concretas del pipeline:

```powershell
python main.py --skip-classification
python main.py --skip-regression
python main.py --skip-clustering
```

## Que Verificar Tras La Ejecucion

Si el pipeline corre correctamente, deberias observar:

- un reporte JSON compacto en consola
- archivos dentro de `outputs/metrics/`
- graficas dentro de `outputs/plots/`

Archivos clave para revisar:

- [`outputs/metrics/pipeline_report.json`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/outputs/metrics/pipeline_report.json)
- [`outputs/metrics/summary.txt`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/outputs/metrics/summary.txt)
- [`outputs/plots/classification_confusion_matrix.png`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/outputs/plots/classification_confusion_matrix.png)
- [`outputs/plots/regression_actual_vs_predicted.png`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/outputs/plots/regression_actual_vs_predicted.png)
- [`outputs/plots/clustering_projection.png`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/outputs/plots/clustering_projection.png)

## Decisiones De Diseno

El proyecto se construyo con estas ideas:

- `main.py` actua como orquestador
- la logica de negocio esta separada por modulos
- cada modelo devuelve resultados en memoria
- el guardado de artefactos se centraliza en un modulo de reporting
- las metricas y los datos de visualizacion se guardan por separado

Esto hace que el proyecto sea mas facil de mantener, probar y ampliar en siguientes fases.

## Proximas Fases

Los siguientes pasos del proyecto son:

- adaptar Jenkins al entorno real donde vaya a ejecutarse

## Reproducibilidad Con Docker

El proyecto ya incluye:

- [`requirements.txt`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/requirements.txt)
- [`Dockerfile`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/Dockerfile)
- [`.dockerignore`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/.dockerignore)

### Construir La Imagen

```powershell
docker build -t sdss-ml-pipeline .
```

### Ejecutar El Pipeline En Docker

```powershell
docker run --rm sdss-ml-pipeline
```

### Ejecutar Y Recuperar Outputs En Tu Maquina

```powershell
docker run --rm -v "${PWD}/outputs:/app/outputs" sdss-ml-pipeline
```

Esto permite que las metricas y graficas generadas dentro del contenedor queden disponibles en tu carpeta local `outputs/`.

## Automatizacion Con Jenkins

El proyecto ya incluye [`Jenkinsfile`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/Jenkinsfile).

### Que Es Un Jenkinsfile

Un `Jenkinsfile` es un archivo de texto que define, paso a paso, que debe hacer Jenkins cuando ejecuta tu proyecto.

Piensalo como una receta de automatizacion:

- de donde sacar el codigo
- como preparar el entorno
- que comandos ejecutar
- como validar que todo salio bien
- que archivos guardar como artefactos

Jenkins lee ese archivo y ejecuta sus etapas automaticamente.

### Que Hace Este Jenkinsfile

El pipeline implementado tiene estas etapas:

1. `Checkout`
   Descarga el codigo del repositorio en el workspace de Jenkins.

2. `Build Docker Image`
   Construye la imagen definida en [`Dockerfile`](c:/Users/nicoh/OneDrive/Dev/Univesidad/BigData/Parcial/Dockerfile).

3. `Run Pipeline In Docker`
   Ejecuta el pipeline dentro del contenedor y monta la carpeta `outputs/` del workspace de Jenkins para recuperar los resultados.

4. `Validate Outputs`
   Verifica que los archivos esperados realmente se hayan generado dentro de `outputs/`.

5. `Archive Artifacts`
   Guarda los archivos de `outputs/` como artefactos del build para poder descargarlos desde Jenkins.

### Como Funciona En La Practica

Cuando Jenkins lanza un build de este proyecto:

1. lee el `Jenkinsfile`
2. ejecuta cada etapa en orden
3. si una etapa falla, detiene el pipeline
4. si todo sale bien, deja los artefactos accesibles desde la interfaz de Jenkins

Eso significa que Jenkins no entrena modelos "por si solo". Lo que hace es automatizar la ejecucion de tu proyecto usando la imagen Docker del proyecto, de forma repetible y visible para un equipo.

### Nota Sobre El Entorno

El `Jenkinsfile` actual esta preparado para agentes Linux o Windows usando comandos `sh` o `bat` segun corresponda.

Requisito importante:

- el agente de Jenkins debe tener Docker disponible

Si mas adelante tu Jenkins corre dentro de contenedores, nodos especificos o credenciales corporativas, el archivo puede ajustarse a ese entorno sin cambiar el pipeline de Python.
