Se han añadido a los archivos que ya existían los nuevos.
- La API que funciona es main_load.py. No obstante no he llegado a entender el error de main_hiper_opt.py
Escogiendo el método de búsqueda bayesiana mencionado, se realiza en mlfow 10 experimentos como prueba.
- Antes de este repositorio se realizaron 50 co un dataset de 9000k ejemplos, pero ante la pérdida de dicho repositorio lo dejo en 10 experiemntos porque tarda demasiado y las diferencias de recall (que es el objetivo a maximizar) no son perceptibles.
- main_load.py: mediante el jupyter notebook modelo_simple_loan donde se utiliza el dataset completo se realiza la optimización de hiperparametros (Al realizar los experimentos, se ve  cual es el mjeor y se seleccionaa ese). Una vez se encunetran los mjeores parametros los he trasladado a main_load y se ha lanzado la api como primera versión.
- main_hiper_opt.py:   Así quería crear una segunda versión de una nueva api que da error y no he podido arreglar. Subo el codigo por si se puede ver el error.
- Dejo los archivos del modelo de deteccion de cancer tambien