import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


st.header("Ejercicio: • Tiempo en horas de funcionamiento de 100 piezas de cierto dispositivo electrónico, transforma los datos para mejorar la simetría.")

# Datos
intervalos = ['[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, 5)', '[5, 6)', '[6, 7)', '[7, 8)', '[8, 9)']
frecuencia = [45, 28, 13, 8, 3, 1, 1, 0, 1]

# Crear un DataFrame
data = {'Intervalo': intervalos, 'Frecuencia': frecuencia}
df = pd.DataFrame(data)

# Configurar el título de la aplicación en Streamlit
st.title('Frecuencia de Piezas por Intervalo de Tiempo con Transformaciones')

# Sidebar
st.sidebar.header('Descripción del Código')
st.sidebar.write("""
Este código permite analizar la distribución de piezas a través de diferentes intervalos de tiempo, y aplicar transformaciones a estos intervalos para observar cómo afectan las frecuencias.

### Funcionalidades:
1. **Visualización Original:** Muestra la frecuencia de piezas en intervalos de tiempo definidos.
2. **Transformaciones:** Permite aplicar distintas transformaciones matemáticas a los intervalos, tales como:
   - \( y = x^2 \)
   - \( y = \sqrt{x} \)
   - \( y = \ln(x) \)
   - \( y = \frac{1}{x} \)
3. **Ajuste de Frecuencias:** Calcula las frecuencias ajustadas para los nuevos intervalos transformados.
4. **Análisis de Distribución:** Compara la asimetría y curtosis de las frecuencias originales y transformadas.
5. **Selección de Mejor Transformación:** Determina cuál transformación mejora la simetría de la distribución.
6. **Visualización Gráfica:** Muestra histogramas de las frecuencias originales y transformadas.

### Cómo Usar:
- Selecciona una transformación en el menú desplegable.
- Haz clic en el botón "Obtener la Mejor Transformación" para ver cuál transformación mejora la simetría de las frecuencias.
- Observa las frecuencias transformadas y los histogramas resultantes para entender el impacto de cada transformación.
""")

# Crear un combobox para que el usuario seleccione la transformación
transformacion = st.selectbox('Selecciona una transformación', ['Original', 'y = x^2', 'y = sqrt(x)', 'y = ln(x)', 'y = 1/x'])

# Función para transformar los intervalos y ajustar las frecuencias
def transformar_intervalos_y_frecuencias(df, transformacion):
    nuevos_intervalos = []
    
    def interseccion_longitud(inf1, sup1, inf2, sup2):
        """ Calcula la longitud de la intersección entre dos intervalos. """
        inf_max = max(inf1, inf2)
        sup_min = min(sup1, sup2)
        return max(0, sup_min - inf_max)

    def transformar_intervalo(inf, sup, transformacion):
        if transformacion == 'y = x^2':
            return inf ** 2, sup ** 2
        elif transformacion == 'y = sqrt(x)':
            return np.sqrt(inf), np.sqrt(sup)
        elif transformacion == 'y = ln(x)':
            return (np.log(inf) if inf > 0 else float('-inf'), 
                    np.log(sup) if sup > 0 else float('-inf'))
        elif transformacion == 'y = 1/x':
            return (1 / sup if sup != 0 else float('inf'), 
                    1 / inf if inf != 0 else float('inf'))
        return inf, sup

    if transformacion == 'Original':
        df_transformado = df.copy()
        df_transformado['Nuevo intervalo'] = df['Intervalo']
        df_transformado['Nueva frecuencia'] = df['Frecuencia']
        return df_transformado, 'Nuevo intervalo', 'Nueva frecuencia'

    for i, intervalo in enumerate(df['Intervalo']):
        inf, sup = map(float, intervalo[1:-1].split(', '))
        nuevo_inf, nuevo_sup = transformar_intervalo(inf, sup, transformacion)
        if np.isinf(nuevo_inf) or np.isinf(nuevo_sup):
            nuevo_intervalo = f'{nuevo_inf}-{nuevo_sup}'
        else:
            nuevo_intervalo = f'{nuevo_inf:.2f}-{nuevo_sup:.2f}'
        nuevos_intervalos.append(nuevo_intervalo)

    # Crear un DataFrame para el nuevo intervalo transformado
    df_transformado = pd.DataFrame({'Nuevo intervalo': nuevos_intervalos})
    
    # Redistribuir las frecuencias
    intervalos_transformados = []
    for x in nuevos_intervalos:
        try:
            intervalos_transformados.append(tuple(map(float, x.split('-'))))
        except ValueError:
            # Manejar casos en que los intervalos no son convertibles a float
            intervalos_transformados.append((float('nan'), float('nan')))

    intervalos_originales = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

    frecuencias_transformadas = []
    
    for int_trans in intervalos_transformados:
        if np.isnan(int_trans[0]) or np.isnan(int_trans[1]) or np.isinf(int_trans[0]) or np.isinf(int_trans[1]):
            frecuencias_transformadas.append(0)
            continue
        
        freq_sum = 0
        for i, int_orig in enumerate(intervalos_originales):
            inf_orig, sup_orig = int_orig
            inf_trans, sup_trans = int_trans
            interseccion_len = interseccion_longitud(inf_orig, sup_orig, inf_trans, sup_trans)
            intervalo_len = sup_orig - inf_orig
            if intervalo_len > 0:
                freq_sum += df['Frecuencia'][i] * (interseccion_len / intervalo_len)
        frecuencias_transformadas.append(freq_sum)

    # Ajustar la suma de frecuencias transformadas para que sea igual a la suma de las frecuencias originales
    suma_frecuencias_originales = df['Frecuencia'].sum()
    suma_frecuencias_transformadas = sum(frecuencias_transformadas)
    if suma_frecuencias_transformadas != 0:
        frecuencias_transformadas = [f * suma_frecuencias_originales / suma_frecuencias_transformadas for f in frecuencias_transformadas]

    # Redondear las frecuencias a enteros
    frecuencias_transformadas = [round(f) for f in frecuencias_transformadas]

    # Asegurar que la suma de frecuencias transformadas sea 100
    suma_frecuencias_transformadas = sum(frecuencias_transformadas)
    if suma_frecuencias_transformadas != 0:
        frecuencias_transformadas = [int(round(f * 100 / suma_frecuencias_transformadas)) for f in frecuencias_transformadas]

    df_transformado['Nueva frecuencia'] = frecuencias_transformadas
    return df_transformado, 'Nuevo intervalo', 'Nueva frecuencia'

# Función para calcular la simetría y la curtosis
def calcular_simetria_y_curtosis(frecuencias):
    return skew(frecuencias), kurtosis(frecuencias)

# Función para determinar la mejor transformación en términos de simetría
def mejor_transformacion(df, transformaciones):
    resultados = {}
    for transformacion in transformaciones:
        df_transformado, _, _ = transformar_intervalos_y_frecuencias(df, transformacion)
        skew_transformado, _ = calcular_simetria_y_curtosis(df_transformado['Nueva frecuencia'])
        resultados[transformacion] = skew_transformado
    mejor = min(resultados, key=resultados.get)
    return mejor, resultados

# Crear un botón para obtener la mejor transformación
if st.button('Obtener la Mejor Transformación'):
    transformaciones = ['Original', 'y = x^2', 'y = sqrt(x)', 'y = ln(x)', 'y = 1/x']
    mejor, resultados = mejor_transformacion(df, transformaciones)
    
    st.write(f'La mejor transformación en términos de simetría es: {mejor}')
    st.write('Asimetría de cada transformación:')
    for t, s in resultados.items():
        st.write(f'{t}: {s:.2f}')

# Mostrar el DataFrame original y el transformado en Streamlit
st.write('Datos Originales de la Frecuencia de Piezas por Intervalo de Tiempo')
st.dataframe(df)

# Aplicar la transformación seleccionada y mostrar los resultados
df_transformado, intervalo_col, frecuencia_col = transformar_intervalos_y_frecuencias(df, transformacion)

st.write('Datos Transformados')
st.dataframe(df_transformado)

# Crear histogramas y calcular asimetría y curtosis
# Frecuencias originales
skew_original, kurtosis_original = calcular_simetria_y_curtosis(df['Frecuencia'])
st.write(f'Asimetría de las frecuencias originales: {skew_original:.2f}')
st.write(f'Curtosis de las frecuencias originales: {kurtosis_original:.2f}')

# Frecuencias transformadas
skew_transformado, kurtosis_transformado = calcular_simetria_y_curtosis(df_transformado['Nueva frecuencia'])
st.write(f'Asimetría de las frecuencias transformadas: {skew_transformado:.2f}')
st.write(f'Curtosis de las frecuencias transformadas: {kurtosis_transformado:.2f}')

# Graficar histogramas
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Histograma de frecuencias originales
ax[0].bar(df['Intervalo'], df['Frecuencia'], color='skyblue')
ax[0].set_title('Histograma de Frecuencias Originales')
ax[0].set_xlabel('Intervalo')
ax[0].set_ylabel('Frecuencia')

# Histograma de frecuencias transformadas
ax[1].bar(df_transformado['Nuevo intervalo'], df_transformado['Nueva frecuencia'], color='lightgreen')
ax[1].set_title(f'Histograma de Frecuencias Transformadas ({transformacion})')
ax[1].set_xlabel('Intervalo')
ax[1].set_ylabel('Frecuencia')

st.pyplot(fig)

# Create 1 buttons that link to Google
if st.button('Calcular nuevas frecuencias'):
    st.write('[Calcular nuevas frecuencias](https://frecuencias-actualizadas-nytfhpgowatameha4hbg5t.streamlit.app/)')


st.sidebar.write("© 2024 Creado por: Javier Horacio Pérez Ricárdez")
