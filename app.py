# -*- coding: utf-8 -*-
"""Predicción de graduación de beneficiarios de becas — Antioquia.

Aplicación Streamlit del Proyecto Integrador de Minería de Datos, UPB 2026.

Captura las características de un beneficiario, reconstruye el vector de entrada
con las mismas transformaciones del entrenamiento y devuelve la probabilidad de
que culmine su programa académico.

Ejecutar con:  streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de graduación · Becas Antioquia",
    page_icon="🎓",
    layout="wide",
)

# ----------------------------------------------------------------- estilos
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

  /* Paleta clara por defecto. Los tokens se redefinen abajo para el tema oscuro,
     de modo que ningun color queda escrito directamente en los componentes. */
  :root {
      --tinta:      #16232b;
      --tinta-2:    #4a5b66;
      --tenue:      #7b8a94;
      --borde:      #dde4e8;
      --superficie: #ffffff;
      --panel:      #f4f7f8;
      --acento:     #1d6b73;
      --acento-2:   #17565c;
      --ok:         #2f6f4e;
      --alerta:     #9a6b12;
      --riesgo:     #a8362b;
      --pista-alta: #d9a49d;
      --pista-med:  #e2c98c;
      --pista-baja: #8fbfa4;
  }

  @media (prefers-color-scheme: dark) {
      :root {
          --tinta:      #e8edf0;
          --tinta-2:    #b3c0c8;
          --tenue:      #8b99a3;
          --borde:      #2d3b44;
          --superficie: #18222a;
          --panel:      #141d24;
          --acento:     #6fb3ba;
          --acento-2:   #8ac7cd;
          --ok:         #68b78c;
          --alerta:     #d5a64a;
          --riesgo:     #e08076;
          --pista-alta: #5d3833;
          --pista-med:  #5a4a24;
          --pista-baja: #2c4d3c;
      }
  }

  html, body, [class*="css"] { font-family: 'Source Sans 3', system-ui, sans-serif; }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }

  .cabecera { border-bottom: 2px solid var(--borde); padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
  .cabecera .sobre {
      font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
      color: var(--acento); font-weight: 600;
  }
  .cabecera h1 {
      font-size: 1.9rem; font-weight: 700; color: var(--tinta);
      margin: .25rem 0 .35rem; line-height: 1.15;
  }
  .cabecera p { color: var(--tinta-2); font-size: 1rem; margin: 0; max-width: 68ch; }

  .grupo {
      font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
      color: var(--tenue); font-weight: 600;
      border-bottom: 1px solid var(--borde); padding-bottom: .4rem; margin-bottom: .9rem;
  }

  .tarjeta {
      border: 1px solid var(--borde); border-radius: 4px; background: var(--superficie);
      padding: 1.6rem 1.8rem; margin-top: .5rem;
  }
  .tarjeta.baja  { border-left: 4px solid var(--ok); }
  .tarjeta.media { border-left: 4px solid var(--alerta); }
  .tarjeta.alta  { border-left: 4px solid var(--riesgo); }

  .veredicto { font-size: 2.6rem; font-weight: 700; line-height: 1.05; letter-spacing: -.02em; }
  .veredicto.baja { color: var(--ok); } .veredicto.media { color: var(--alerta); } .veredicto.alta { color: var(--riesgo); }
  .veredicto-pie { color: var(--tinta-2); font-size: 1rem; margin-top: .45rem; }
  .veredicto-pie b { color: var(--tinta); }

  .confianza {
      display: inline-block; margin-top: .9rem; padding: .3rem .7rem; border-radius: 3px;
      background: var(--panel); border: 1px solid var(--borde);
      color: var(--tinta-2); font-size: .87rem;
  }
  .confianza b { color: var(--tinta); }

  .accion { color: var(--tinta-2); font-size: .97rem; line-height: 1.55; }

  .pista { position: relative; height: 10px; border-radius: 5px; margin: 1.4rem 0 .5rem;
           background: linear-gradient(to right,
              var(--pista-alta) 0%, var(--pista-alta) 35%,
              var(--pista-med) 35%, var(--pista-med) 65%,
              var(--pista-baja) 65%, var(--pista-baja) 100%); }
  .marca { position: absolute; top: -5px; width: 3px; height: 20px;
           background: var(--tinta); border-radius: 2px; }
  .escala { display: flex; justify-content: space-between; font-size: .73rem; color: var(--tenue); }

  .aviso { color: var(--tenue); font-size: .84rem; line-height: 1.5;
           border-top: 1px solid var(--borde); margin-top: 1.4rem; padding-top: .9rem; }

  div[data-testid="stForm"] { border: 1px solid var(--borde); border-radius: 4px; padding: 1.6rem 1.8rem; }
  div[data-testid="stFormSubmitButton"] button {
      background: var(--acento); color: #fff; border: none; font-weight: 600;
      padding: .6rem 0; border-radius: 3px;
  }
  div[data-testid="stFormSubmitButton"] button:hover { background: var(--acento-2); color: #fff; }
  section[data-testid="stSidebar"] { border-right: 1px solid var(--borde); }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ modelo
# El orden de la lista es el contrato con el cuaderno de clasificación, que la
# serializa como [modelo, label_encoder, variables, scaler].
@st.cache_resource
def cargar_modelo(ruta='modelo-clasificacion.pkl'):
    with open(ruta, 'rb') as f:
        return pickle.load(f)

modelo, label_encoder, variables, min_max_scaler = cargar_modelo()


# ------------------------------------------------------------- catálogos
programas_cursados_lista = ['ADMINISTRACION DE EMPRESAS',
 'ADMINISTRACION DE EMPRESAS AGROPECUARIAS',
 'ADMINISTRACION DE EMPRESAS TURISTICAS Y HOTELERAS',
 'ADMINISTRACION EN SEGURIDAD Y SALUD EN EL TRABAJO',
 'ADMINISTRACION FINANCIERA',
 'AGRONOMIA',
 'ANTROPOLOGIA',
 'ARQUITECTURA',
 'ARTES PLASTICAS',
 'BIOINGENIERIA',
 'BIOLOGIA',
 'CICLO COMPLEMENTARIO',
 'CIENCIA POLITICA',
 'CIENCIAS AMBIENTALES',
 'COMERCIO EXTERIOR',
 'COMUNICACION AUDIOVISUAL',
 'COMUNICACION SOCIAL',
 'COMUNICACION SOCIAL - PERIODISMO',
 'COMUNICACION SOCIAL Y PERIODISMO',
 'CONSTRUCCION',
 'CONTADURIA',
 'CONTADURIA PUBLICA',
 'DERECHO',
 'DESARROLLO TERRITORIAL',
 'ECOLOGIA DE ZONAS COSTERAS',
 'ECONOMIA',
 'ENFERMERIA',
 'ENTRENAMIENTO DEPORTIVO',
 'ESTADISTICA',
 'FILOLOGIA HISPANICA',
 'FISICA',
 'GERONTOLOGIA',
 'GESTION ADMINISTRATIVA',
 'GESTION CULTURAL',
 'GESTION EN ECOLOGIA Y TURISMO',
 'HISTORIA',
 'INGENIERIA ADMINISTRATIVA',
 'INGENIERIA AGRICOLA',
 'INGENIERIA AGROINDUSTRIAL',
 'INGENIERIA AGRONOMICA',
 'INGENIERIA AGROPECUARIA',
 'INGENIERIA AMBIENTAL',
 'INGENIERIA BIOLOGICA',
 'INGENIERIA BIOMEDICA',
 'INGENIERIA BIOQUIMICA',
 'INGENIERIA CIVIL',
 'INGENIERIA DE ALIMENTOS',
 'INGENIERIA DE CONTROL',
 'INGENIERIA DE MATERIALES',
 'INGENIERIA DE MINAS Y METALURGIA',
 'INGENIERIA DE PRODUCTIVIDAD Y CALIDAD',
 'INGENIERIA DE SISTEMAS',
 'INGENIERIA DE SISTEMAS E INFORMATICA',
 'INGENIERIA DE TELECOMUNICACIONES',
 'INGENIERIA ELECTRICA',
 'INGENIERIA ELECTRONICA',
 'INGENIERIA EN HIGIENE Y SEGURIDAD OCUPACIONAL',
 'INGENIERIA EN SISTEMAS E INFORMATICA',
 'INGENIERIA EN SOFTWARE',
 'INGENIERIA FISICA',
 'INGENIERIA FORESTAL',
 'INGENIERIA GEOLOGICA',
 'INGENIERIA INDUSTRIAL',
 'INGENIERIA INFORMATICA',
 'INGENIERIA MECANICA',
 'INGENIERIA MECATRONICA',
 'INGENIERIA OCEANOGRAFICA',
 'INGENIERIA QUIMICA',
 'INGENIERIA SANITARIA',
 'INGENIERIA URBANA',
 'INSTRUMENTACION QUIRURGICA',
 'LIC. ARTES PLASTICAS',
 'LIC. EN EDUCACION BASICA CON ENFASIS EN CIENCIAS NATURALES Y EDUCACION AMBIENTAL',
 'LIC. EN EDUCACION BASICA CON ENFASIS EN EDUCACION ARTISTICA Y CULTURAL',
 'LIC. EN EDUCACION BASICA CON ENFASIS EN HUMANIDADES, LENGUA CASTELLANA',
 'LIC. EN EDUCACION ESPECIAL',
 'LIC. EN EDUCACION PREESCOLAR',
 'LIC. EN FILOSOFIA',
 'LIC. EN LENGUAS EXTRANJERAS',
 'LIC. EN MATEMATICAS Y FISICA',
 'LIC. EN PEDAGOGIA INFANTIL',
 'LICENCIATURA EN CIENCIAS SOCIALES',
 'LICENCIATURA EN EDUCACION BASICA CON ENFASIS EN HUMANIDADES, LENGUA CASTELLANA',
 'LICENCIATURA EN EDUCACION BASICA ENFASIS EN EDUCACION FISICA, RECREACION Y DEPORTE',
 'LICENCIATURA EN EDUCACION ESPECIAL',
 'LICENCIATURA EN EDUCACION FISICA',
 'LICENCIATURA EN EDUCACION FISICA Y DEPORTE',
 'LICENCIATURA EN EDUCACION INFANTIL',
 'LICENCIATURA EN EDUCACION PREESCOLAR',
 'LICENCIATURA EN LENGUAS EXTRANJERAS',
 'LICENCIATURA EN LENGUAS EXTRANJERAS CON ENFASIS EN INGLES',
 'LICENCIATURA EN LITERATURA Y LENGUA CASTELLANA',
 'LICENCIATURA EN PEDAGOGIA DE LA MADRE TIERRA',
 'LICENCIATURA EN PEDAGOGIA INFANTIL',
 'MAESTRIA EN EDUCACION',
 'MATEMATICAS',
 'MEDICINA',
 'MEDICINA VETERINARIA',
 'MICROBIOLOGIA INDUSTRIAL Y AMBIENTAL',
 'MICROBIOLOGIA Y BIOANALISIS',
 'NEGOCIOS INTERNACIONALES',
 'NUTRICION Y DIETETICA',
 'OCEANOGRAFIA',
 'ODONTOLOGIA',
 'OTRO',
 'PROFESIONAL EN DEPORTE',
 'PROFESIONAL EN PSICOLOGIA',
 'PROFESIONAL EN TRABAJO SOCIAL',
 'PSICOLOGIA',
 'PUBLICIDAD Y MERCADEO DIGITAL',
 'SOCIOLOGIA',
 'TECNICA PROFESIONAL EN ATENCION PREHOSPITALARIA',
 'TECNICA PROFESIONAL EN PROCESOS EMPRESARIALES  RURALES',
 'TECNICA PROFESIONAL EN PROCESOS EMPRESARIALES RURALES',
 'TECNICA PROFESIONAL EN PROCESOS RURALES EMPRESARIALES',
 'TECNICA PROFESIONAL EN REDES ELECTRICAS DE DISTRIBUCION DE ENERGIA CON CICLO PROPEDEUTICO A LA TECNOLOGIA EN SUPERVISION DE DISTRIBUCION DE REDES',
 'TECNICO PROFESIONAL EN ATENCION PREHOSPITALARIA',
 'TECNICO PROFESIONAL EN REDES ELECTRICAS DE DISTRIBUCION DE ENERGIA',
 'TECNICO PROFESIONAL EN SOP. DE SISTEMAS DE INFORMACION',
 'TECNOLOGIA AGROAMBIENTAL',
 'TECNOLOGIA AGROPECUARIA',
 'TECNOLOGIA DE ALIMENTOS',
 'TECNOLOGIA ELECTRONICA',
 'TECNOLOGIA EN ANALISIS Y DESARROLLO DE SISTEMAS DE INFORMACION',
 'TECNOLOGIA EN ARCHIVISTICA',
 'TECNOLOGIA EN CONSTRUCCIONES CIVILES',
 'TECNOLOGIA EN CONTABILIDAD Y FINANZAS',
 'TECNOLOGIA EN CONTROL AMBIENTAL',
 'TECNOLOGIA EN DESARROLLO DE SOFTWARE',
 'TECNOLOGIA EN DISENO  Y DESARROLLO DE SISTEMAS DE INFORMACION',
 'TECNOLOGIA EN DISENO Y DESARROLLO DE SISTEMAS DE INFORMACION',
 'TECNOLOGIA EN ECOLOGIA Y TURISMO',
 'TECNOLOGIA EN ENTRENAMIENTO DEPORTIVO',
 'TECNOLOGIA EN GESTION ADMINISTRATIVA',
 'TECNOLOGIA EN GESTION AGROPECUARIA',
 'TECNOLOGIA EN GESTION BANCARIA Y DE ENTIDADES FINANCIERAS',
 'TECNOLOGIA EN GESTION CATASTRAL Y AGRIMENSURA',
 'TECNOLOGIA EN GESTION DE AGROINDUSTRIAS ALIMENTICIAS',
 'TECNOLOGIA EN GESTION DE CALIDAD EN ALIANZA CON EL ICONTEC',
 'TECNOLOGIA EN GESTION DE EMPRESAS AGROPECUARIAS',
 'TECNOLOGIA EN GESTION DE LA SEGURIDAD Y SALUD EN EL TRABAJO',
 'TECNOLOGIA EN GESTION DE NEGOCIOS',
 'TECNOLOGIA EN GESTION DE PLANTACIONES FORESTALES',
 'TECNOLOGIA EN GESTION DE PROCESOS ADMINISTRATIVOS EN SALUD',
 'TECNOLOGIA EN GESTION DE RECURSOS NATURALES',
 'TECNOLOGIA EN GESTION DE SERVICIOS DE SALUD',
 'TECNOLOGIA EN GESTION DEL TALENTO HUMANO',
 'TECNOLOGIA EN GESTION DEL TURISMO DE NATURALEZA',
 'TECNOLOGIA EN GESTION INDUSTRIAL',
 'TECNOLOGIA EN GESTION LOGISTICA INTEGRAL',
 'TECNOLOGIA EN GESTION PUBLICA',
 'TECNOLOGIA EN LOGISTICA',
 'TECNOLOGIA EN LOGISTICA INTEGRAL',
 'TECNOLOGIA EN LOGISTICA Y MERCADEO INTERNACIONAL',
 'TECNOLOGIA EN MANTENIMIENTO DE EQUIPOS DE COMPUTO DISENO E INSTALACION DE CABLEADO ESTRUCTURADO',
 'TECNOLOGIA EN MANTENIMIENTO DE EQUIPOS DE COMPUTO, DISENO E INSTALACION DE CABLEADO ESTRUCTURADO',
 'TECNOLOGIA EN MECANICA AUTOMOTRIZ',
 'TECNOLOGIA EN MECANICA INDUSTRIAL',
 'TECNOLOGIA EN PRODUCCION AGROECOLOGICA',
 'TECNOLOGIA EN PRODUCCION AGROINDUSTRIAL DE ALIMENTOS (LINEAS CARNICOS, LACTEOS Y FRUVER)',
 'TECNOLOGIA EN PRODUCCION AGROPECUARIA',
 'TECNOLOGIA EN REGENCIA DE FARMACIA',
 'TECNOLOGIA EN SANEAMIENTO AMBIENTAL',
 'TECNOLOGIA EN SEGURIDAD E HIGIENE OCUPACIONAL',
 'TECNOLOGIA EN SISTEMAS DE GESTION AMBIENTAL',
 'TECNOLOGIA EN SISTEMAS DE INFORMACION',
 'TECNOLOGIA EN SISTEMAS DE INFORMACION EN SALUD',
 'TECNOLOGIA EN SISTEMAS MECATRONICOS',
 'TECNOLOGIA EN SUPERVISION DE REDES DE DISTRIBUCION DE ENERGIA ELECTRICA',
 'TECNOLOGIA EN SUPERVISION DE SISTEMAS DE GENERACION Y DISTRIBUCION DE ENERGIA ELECTRICA',
 'TECNOLOGIA EN SUPERVISION DE SISTEMAS ELECTRICOS DE POTENCIA',
 'TECNOLOGIA EN TELECOMUNICACIONES',
 'TECNOLOGIA MECANICA INDUSTRIAL',
 'TRABAJO SOCIAL',
 'ZOOTECNIA']

v_semestre = [1, 2]
v_beneficio = ['MATRICULA', 'MATRICULA Y SOSTENIMIENTO', 'SOSTENIMIENTO']
v_genero = ['FEMENINO', 'MASCULINO']
v_estrato = ['ESTRATO 1', 'ESTRATO 2', 'ESTRATO 3', 'ESTRATO 4', 'ESTRATO 5']
v_etnia = ['AFROCOLOMBIANO', 'INDIGENA', 'NINGUNO']
v_victima = ['NO', 'SI']
v_subregiones = ['BAJO CAUCA', 'MAGDALENA MEDIO', 'NORDESTE', 'NORTE', 'OCCIDENTE',
                 'ORIENTE', 'SUROESTE', 'URABA', 'VALLE DE ABURRA']
v_programas_cursados = programas_cursados_lista
v_universidades = ['COLEGIO MAYOR DE ANTIOQUIA',
 'COREDI',
 'CORPORACION UNIVERSITARIA MINUTO DE DIOS -UNIMINUTO-',
 'CORPORACION UNIVERSITARIA REMINGTON',
 'ESCUELA NORMAL SUPERIOR MARIA AUXILIADORA',
 'ESCUELA NORMAL SUPERIOR RAFAEL MARIA GIRALDO',
 'ESCUELA NORMAL SUPERIOR SAN PEDRO DE LOS MILAGROS',
 'ESCUELA SUPERIOR TECNOLOGICA DE ARTES DEBORA ARANGO',
 'FUNDACION DE ESTUDIOS SUPERIORES UNIVERSITARIOS DE URABA ANTONIO ROLDAN BETANCUR',
 'FUNDACION UNIVERSITARIA CATOLICA DEL NORTE',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DE ABEJORRAL',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DE AMAGA',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DE JERICO',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DEL BAJO CAUCA',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DEL MAGDALENA MEDIO',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR DEL NORDESTE',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR GENOVEVA DIAZ',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR PEDRO JUSTO BERRIO',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR SAGRADA FAMILIA',
 'INSTITUCION EDUCATIVA ESCUELA NORMAL SUPERIOR SANTA TERESITA',
 'INSTITUCION UNIVERSITARIA DE ENVIGADO',
 'INSTITUCION UNIVERSITARIA DIGITAL DE ANTIOQUIA',
 'INSTITUCION UNIVERSITARIA MARCO FIDEL SUAREZ - IUMAFIS',
 'INSTITUCION UNIVERSITARIA PASCUAL BRAVO',
 'INSTITUCION UNIVERSITARIA POLITECNICO GRANCOLOMBIANO',
 'INSTITUTO TECNOLOGICO METROPOLITANO',
 'INSTIUCION EDUCATIVA NORMAL SUPERIOR SAN ROQUE',
 'OTRO',
 'POLITECNICO COLOMBIANO JAIME ISAZA CADAVID',
 'SENA',
 'TECNOLOGICO DE ANTIOQUIA',
 'TECOC',
 'UNIVERSIDAD CATOLICA DE ORIENTE',
 'UNIVERSIDAD DE ANTIOQUIA',
 'UNIVERSIDAD DE MEDELLIN',
 'UNIVERSIDAD DE SAN BUENAVENTURA',
 'UNIVERSIDAD EIA',
 'UNIVERSIDAD NACIONAL ABIERTA Y A DISTANCIA',
 'UNIVERSIDAD NACIONAL DE COLOMBIA',
  'UNIVERSIDAD PONTIFICIA BOLIVARIANA']
v_formacion = ['NORMALISTA', 'POSTGRADO', 'TECNICA PROFESIONAL', 'TECNOLOGICA', 'UNIVERSITARIA']
v_movilidad = [False, True]
v_am = ['NO', 'SI']
v_am_oferta = ['NO', 'SI']

AREA_METROPOLITANA = ("Medellín, Barbosa, Girardota, Copacabana, Bello, Envigado, "
                      "Itagüí, Sabaneta, La Estrella o Caldas.")


# ----------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("#### El modelo")
    st.caption("Votación blanda sobre cinco familias: regresión logística, SVM, "
               "red neuronal, Random Forest y XGBoost.")
    izq, der = st.columns(2)
    izq.metric("Exactitud", "72,1 %")
    der.metric("AUC-ROC", "0,795")

    with st.expander("Cómo leer el resultado"):
        st.markdown(
            "El resultado trae el veredicto y la probabilidad que lo sustenta. La "
            "confianza **no es la misma en todo el rango**: medido sobre el conjunto de "
            "prueba, el modelo acierta el 84 % cuando estima menos de 0,35 y el 78 % "
            "cuando estima más de 0,65, pero solo el **58 %** en la franja intermedia, "
            "donde cae el 38 % de los casos. Por eso sirve para **priorizar "
            "acompañamiento**, no para negar, condicionar o retirar un beneficio."
        )

    with st.expander("Limitaciones"):
        st.markdown(
            "**Etiqueta censurada en el tiempo.** La tasa de graduación pasa de 67,9 % "
            "en la convocatoria de 2013 a 0 % en las de 2023 y 2024: quien recibió el "
            "apoyo hace poco no ha tenido tiempo de graduarse. Parte de lo que el modelo "
            "lee como riesgo es antigüedad de la cohorte.\n\n"
            "**Variables sensibles.** Género, estrato, grupo étnico y condición de víctima "
            "entran al modelo, y las tasas difieren de forma marcada entre grupos.\n\n"
            "**Sin grupo de comparación.** Los datos solo contienen beneficiarios, así que "
            "no puede estimarse el efecto del auxilio."
        )

    st.caption("Datos: Corporación Gilberto Echeverri Mejía · 14.552 beneficiarios · "
               "Datos Abiertos Colombia.")


# --------------------------------------------------------------- cabecera
st.markdown("""
<div class="cabecera">
  <div class="sobre">Aprendizaje de máquinas</div>
  <h1>Riesgo de no graduación en beneficiarios de apoyos económicos</h1>
  <p>Estima la probabilidad de que un beneficiario de beca o crédito condonable
     en Antioquia culmine su programa académico, para priorizar acompañamiento
     temprano.</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------- formulario
with st.form("prediccion"):
    izquierda, derecha = st.columns(2, gap="large")

    with izquierda:
        st.markdown('<div class="grupo">Perfil de la persona</div>', unsafe_allow_html=True)
        edad = st.slider('Edad al momento de la convocatoria', 13, 70, 20)
        genero = st.selectbox('Género', v_genero)
        estrato = st.selectbox('Estrato socioeconómico', v_estrato,
                               help="Estrato de la vivienda. Los datos solo contienen del 1 al 5.")
        etnia = st.selectbox('Grupo étnico', v_etnia,
                             help="Autorreconocimiento étnico declarado en la inscripción.")
        victima = st.selectbox('¿Es víctima del conflicto armado?', v_victima)
        # El modelo espera el booleano original (la columna entrenada es
        # movilidad_territorial_True); format_func solo cambia lo que ve el usuario.
        movilidad = st.selectbox('¿Reside fuera de su departamento de nacimiento?', v_movilidad,
                                 format_func=lambda x: 'SÍ' if x else 'NO',
                                 help="Movilidad territorial: indica desarraigo respecto del "
                                      "departamento de origen.")
        subregiones = st.selectbox('Subregión de residencia', v_subregiones,
                                   help="Una de las nueve subregiones de Antioquia.")
        residencia_am = st.selectbox('¿Reside en el Área Metropolitana?', v_am,
                                     help="Municipios del Valle de Aburrá: " + AREA_METROPOLITANA)

    with derecha:
        st.markdown('<div class="grupo">Contexto académico</div>', unsafe_allow_html=True)
        semestre = st.selectbox('Semestre de la convocatoria', v_semestre,
                                help="Ciclo académico en el que aplicó al beneficio.")
        beneficio = st.selectbox('Beneficio otorgado', v_beneficio,
                                 help="Matrícula, sostenimiento o ambos.")
        tipo_formacion = st.selectbox('Tipo de formación', v_formacion,
                                      help="Nivel del programa que va a cursar.")
        universidad = st.selectbox('Institución de educación superior', v_universidades)
        programas_cursados = st.selectbox('Programa académico', v_programas_cursados)
        am_oferta = st.selectbox('¿El programa se oferta en el Área Metropolitana?', v_am_oferta,
                                 help="Municipio donde se dicta el programa: " + AREA_METROPOLITANA)

    st.write("")
    enviar = st.form_submit_button("Calcular probabilidad", use_container_width=True)


# --------------------------------------------------------------- resultado
if enviar:
    # Diccionario con los nombres de las columnas del CSV original
    input_data = {
        'SEMESTRE DE CONVOCATORIA': semestre,
        'BENEFICIO OTORGADO': beneficio,
        'GENERO': genero,
        'ESTRATO': estrato,
        'GRUPO ETNICO': etnia,
        'VICTIMA DEL CONFLICTO ARMADO': victima,
        'UNIVERSIDAD': universidad,
        'TIPO DE FORMACION': tipo_formacion,
        'edad_beneficiario': edad,
        'movilidad_territorial': movilidad,
        'Municipio_Residencia_Area_metropolitana': residencia_am,
        'SUBREGION DE RESIDENCIA': subregiones,
        'PROGRAMA CURSADO': programas_cursados,
        'Municipio_Oferta_Area_metropolitana': am_oferta
    }
    data = pd.DataFrame([input_data])

    # Se replica la preparación del entrenamiento. En despliegue drop_first=False:
    # se generan todos los niveles y el reindex se queda con las 247 columnas del
    # modelo, descartando los niveles de referencia que allá se habían eliminado.
    data_preparada = data.copy()
    data_preparada[['edad_beneficiario']] = min_max_scaler.transform(
        data_preparada[['edad_beneficiario']])
    data_preparada = pd.get_dummies(
        data_preparada,
        columns=['PROGRAMA CURSADO', 'SUBREGION DE RESIDENCIA', 'BENEFICIO OTORGADO',
                 'ESTRATO', 'TIPO DE FORMACION', 'SEMESTRE DE CONVOCATORIA', 'GENERO',
                 'movilidad_territorial', 'VICTIMA DEL CONFLICTO ARMADO', 'GRUPO ETNICO',
                 'UNIVERSIDAD', 'Municipio_Residencia_Area_metropolitana',
                 'Municipio_Oferta_Area_metropolitana'],
        drop_first=False, dtype=int)
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    probabilidad = float(modelo.predict_proba(data_preparada)[0, 1])
    porcentaje = probabilidad * 100
    se_gradua = bool(modelo.predict(data_preparada)[0])

    # La confianza no es la misma en todo el rango. Medido sobre los 4.366 registros
    # del conjunto de prueba, el modelo acierta el 84 % por debajo de 0,35 y el 78 %
    # por encima de 0,65, pero solo el 58 % en la franja intermedia, donde de hecho
    # se graduo el 52 % de los casos: ahi su prediccion no es informativa.
    if probabilidad >= 0.65:
        clase, banda, acierto = "baja", "Riesgo bajo", "78 %"
        accion = ("El perfil se parece al de quienes culminaron. Seguimiento estándar, "
                  "sin medidas adicionales.")
    elif probabilidad >= 0.35:
        clase, banda, acierto = "media", "Zona de incertidumbre", "58 %"
        accion = ("En esta franja el modelo apenas supera al azar: acierta 56 de cada 100 "
                  "veces, y de las personas que caen aquí se graduó el 52 %. Trate la "
                  "predicción como no concluyente y decida con información que el modelo "
                  "no ve. Es el 38 % de los beneficiarios.")
    else:
        clase, banda, acierto = "alta", "Riesgo alto", "84 %"
        accion = ("El perfil se parece al de quienes no culminaron. Priorizar acompañamiento "
                  "académico y seguimiento desde el primer semestre.")

    veredicto = "SÍ se gradúa" if se_gradua else "NO se gradúa"

    st.markdown(f"""
    <div class="tarjeta {clase}">
      <div class="veredicto {clase}">{veredicto}</div>
      <div class="veredicto-pie">Probabilidad estimada de culminar el programa:
        <b>{porcentaje:.1f} %</b> &nbsp;·&nbsp; {banda}</div>
      <div class="confianza">En este rango el modelo acierta el <b>{acierto}</b> de las veces</div>
      <div class="pista"><div class="marca" style="left: calc({porcentaje:.1f}% - 1.5px);"></div></div>
      <div class="escala"><span>0 % · riesgo alto</span><span>35 %</span><span>65 %</span><span>riesgo bajo · 100 %</span></div>
      <div class="accion" style="margin-top:1.1rem;">{accion}</div>
      <div class="aviso">El modelo acierta en el 72,1 % de los casos en promedio, pero esa
        cifra esconde tres regímenes distintos: 84 % por debajo de 0,35, 58 % entre 0,35 y
        0,65, y 78 % por encima. Esta estimación orienta la priorización de acompañamiento;
        no debe usarse para negar, condicionar o retirar un beneficio.</div>
    </div>
    """, unsafe_allow_html=True)


else:
    st.info("Complete el perfil y pulse **Calcular probabilidad**.")
