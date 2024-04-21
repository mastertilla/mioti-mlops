## Descripción
Esta API utiliza un modelo de regresión logística para predecir decisiones de hipotecas basadas en datos financieros personales. Implementada con FastAPI y protegida con SlowAPI para limitar la cantidad de solicitudes y evitar el abuso del servicio.

## Entradas del modelo: 
```Ingresos```: los ingresos de la familia mensual
- ```Gastos comunes```: pagos de luz, agua, gas, etc mensual
- ```Pago coche```: si se está pagando cuota por uno o más coches, y los gastos en combustible, etc al mes.
- ```Gastos_otros```: compra en supermercado y lo necesario para vivir al mes
- ```Ahorros```: suma de ahorros dispuestos a usar para la compra de la casa.
- ```Vivienda```: precio de la vivienda que quiere comprar esa familia
- ```Estado civil```:
    -   0-soltero
    -   1-casados
    -   2-divorciados
- ```Hijos```: cantidad de hijos menores y que no trabajan.
- ```Trabajo```:
    -   0-sin empleo 
    -   1-autónomo (freelance)
    -   2-empleado
    -   3-empresario
    -   4-pareja: autónomos
    -   5-pareja: empleados
    -   6-pareja: autónomo y asalariado
    -   7-pareja:empresario y autónomo
    -   8-pareja: empresarios los dos o empresario y empleado
- ```Comprar```: 
    - 0-No comprar 
    - 1-Comprar (_esta será nuestra columna de salida, para aprender_)

## Ejemplo de solicitud
{
  "ingresos": 5000,
  "gastos_comunes": 1000,
  "pago_coche": 500,
  "gastos_otros": 700,
  "ahorros": 20000,
  "vivienda": 300000,
  "estado_civil": 1,
  "hijos": 2,
  "trabajo": 1