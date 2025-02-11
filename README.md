# 📚 TFG: Evaluación de la Recursividad en LLMs

Este repositorio contiene el código, datasets y resultados de experimentos realizados durante el TFG, cuyo objetivo es analizar el impacto de la recursividad en la generación de texto por parte de modelos de lenguaje de gran tamaño (LLMs).

## 📂 Estructura del repositorio
📦 TFG_Sandra<br>
├── 📁 Datasets. Conjunto de datos utilizados en los experimentos<br>
├── 📁Resultados. Resultados generados a partir de las pruebas con los modelos <br>
  └── 📁MMLU<br>
    └── 📁Preguntas extensas N iteraciones. Contiene los resultados de ejecutar varias iteraciones recursivas.<br>
├── 📁 scripts. Código para preprocesamiento, evaluación y experimentos (próximamente) <br>
└── README.md. Documentación del repositorio<br>

## 📄 Descripción del proyecto

El estudio se centra en evaluar cómo la información se degrada a través de procesos iterativos de reformulación y parafraseo en modelos de lenguaje. Se analizan métricas de similitud, tasa de aciertos y aparición de alucinaciones en los modelos.

## 📊 Resultados obtenidos

- Se han realizado pruebas con preguntas del dataset MMLU en diferentes niveles de reformulación.
- Se ha observado que a partir de *N* iteraciones, la precisión del modelo disminuye significativamente.

📍 **Ver detalles en la carpeta [`Resultados/`](./Resultados/).**

## 📁 Datasets utilizados

Los conjuntos de datos empleados en este estudio incluyen:

- **MMLU**: Conjunto de preguntas del Massive Multitask Language Understanding (MMLU). 
- **MMLU - History Questions**: Subconjunto de preguntas del Massive Multitask Language Understanding (MMLU). 
- **Parafraseos generados**: Preguntas reformuladas mediante LLMs a lo largo de iteraciones sucesivas.

📍 **Disponibles en la carpeta [`Datasets/`](./Datasets/).**

## 🚀 Próximos pasos

🔹 Incorporación de scripts para la automatización del proceso de parafraseo y evaluación.  
🔹 Expansión del análisis a otros datasets y modelos.  
🔹 Implementación de visualizaciones interactivas para los resultados.  

## 🛠 Requisitos y configuración

⚙️ **Dependencias necesarias** *(se agregarán en el futuro)*  
⚡ **Instrucciones para la ejecución** *(pendiente de añadir)*  
  


---

✉ **Contacto:** Para dudas o colaboración, puedes abrir un *issue* en este repositorio.  
📢 **Actualizaciones:** Se agregarán nuevas secciones conforme el proyecto avance.



