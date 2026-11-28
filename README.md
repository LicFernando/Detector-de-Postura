# Monitor Inteligente de Postura en Tiempo Real

Sistema de visión por computadora desarrollado en Python con **MediaPipe (`PoseLandmarker`)**, **OpenCV** y **Matplotlib** para monitorear la postura cervical y dorsal frente al monitor en tiempo real, sin entrenar redes neuronales desde cero.

## Características Principales
* **Calibración Híbrida:** Combina márgenes de referencia extraídos de imágenes de muestra (`muestras/normal/` y `muestras/encorvado/`) con una calibración rápida personalizada en vivo para adaptarse a la complexión, altura de silla y ángulo de cámara de cada usuario.
* **Detección Geométrica Suavizada:** Evalúa tanto la inclinación cervical-dorsal (`Ang`) como el hundimiento vertical normalizado (`Caida`) de cabeza y hombros, aplicando suavizado móvil y validación por frames consecutivos para evitar falsos positivos.
* **Alertas Silenciosas y Anti-Spam:** Muestra notificaciones emergentes no bloqueantes (`tkinter`) sin audio cuando se detecta encorvamiento sostenido, reguladas por un tiempo de enfriamiento (*cooldown*) y límite de frecuencia por ventana de tiempo.
* **Reporte y Gráfica Estadística Automáticos:** Exporta al cerrar la sesión un resumen cronológico en `.txt` y una gráfica comparativa en `.png` dentro de la carpeta `reportes/`.

## Estructura del Proyecto
```text
monitor-postura-ia/
├── muestras/
│   ├── normal/          # Imágenes (.jpg, .png) de postura erguida frente al monitor
│   └── encorvado/       # Imágenes (.jpg, .png) de encorvamiento gradual en la silla
├── reportes/            # Se crea automáticamente al finalizar con reporte .txt y gráfica .png
├── .gitignore
├── monitor_postura.py
├── pose_landmarker_lite.task
├── README.md
└── requirements.txt

INSTALACION Y EJECUCION

## 1. Clonar Repositorio

git clone https://github.com/LicFernando/Detector-de-Postura.git
cd Detector-de-Postura

## 2. Crear y activar el entorno virtual (Python 3.11 o 3.12)

#Windows

py -3.12 -m venv .venv
.\.venv\Scripts\activate

#En Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

## 3. Instalar dependencias:

pip install -r requirements.txt

## 4. Ejecutar en el monitor

python monitor_postura.py

