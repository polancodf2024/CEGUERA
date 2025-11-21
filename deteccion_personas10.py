# deteccion_personas_streamlit_cloud.py
import cv2
import numpy as np
import time
import pygame
import os
import streamlit as st
from PIL import Image
import tempfile

class SistemaDeteccionPersonasStreamlit:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.frame_actual = None
        self.ultima_alerta = 0
        self.cooldown = 2.0
        
        # En Cloud, deshabilitar audio y cámara
        self.audio_disponible = False
        self.camara_disponible = False
        self.sonidos = {}
        
        print("✅ Sistema Streamlit Cloud inicializado")
        
    def inicializar_deteccion_avanzada(self):
        """Inicializa detectores sin GUI"""
        print("🔍 Inicializando detectores...")
        self.detectores = []
        
        # Haar Cascade para cuerpo completo
        try:
            haar_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(haar_path):
                detector_haar = cv2.CascadeClassifier(haar_path)
                if not detector_haar.empty():
                    self.detectores.append(('haar_fullbody', detector_haar))
                    print("✅ Haar Fullbody cargado")
        except Exception as e:
            print(f"⚠️ Error Haar Fullbody: {e}")
        
        # Detección por movimiento
        self.frame_anterior = None
        self.detectores.append(('movimiento', None))
        print("✅ Detección por movimiento disponible")
        
    def configurar_parametros(self):
        """Configura parámetros del sistema"""
        self.umbral_confianza = 0.3
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        self.calibracion_distancia = {
            'ancho_referencia_cerca': 250,
            'ancho_referencia_lejos': 80,
            'distancia_referencia_cerca': 0.4,
            'distancia_referencia_lejos': 2.0,
            'factor_ajuste_camara': 1.2
        }
        
        self.zonas = {
            'izquierda': (0, self.ancho_pantalla * 0.4),
            'frente': (self.ancho_pantalla * 0.4, self.ancho_pantalla * 0.6),
            'derecha': (self.ancho_pantalla * 0.6, self.ancho_pantalla)
        }

    def detectar_con_haar(self, frame, detector, nombre):
        """Detección usando Haar Cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if 'fullbody' in nombre:
                min_size = (30, 60)
                scale_factor = 1.05
            else:
                min_size = (30, 30)
                scale_factor = 1.1
                
            detecciones = detector.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=3,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            resultados = []
            for (x, y, w, h) in detecciones:
                centro_x = x + w//2
                centro_y = y + h//2
                
                resultados.append({
                    'bbox': (x, y, x + w, y + h),
                    'centro': (centro_x, centro_y),
                    'confianza': 0.7,
                    'distancia_estimada': self.estimar_distancia_mejorada(w, h),
                    'detector': nombre
                })
                
            return resultados
        except Exception as e:
            print(f"❌ Error en Haar: {e}")
            return []

    def detectar_por_movimiento(self, frame):
        """Detección por movimiento"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.frame_anterior is None:
                self.frame_anterior = gray
                return []
            
            frame_diff = cv2.absdiff(self.frame_anterior, gray)
            thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            resultados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contorno)
                    
                    relacion = h / w if w > 0 else 0
                    if 1.2 < relacion < 4.0:
                        centro_x = x + w//2
                        
                        resultados.append({
                            'bbox': (x, y, x + w, y + h),
                            'centro': (centro_x, y + h//2),
                            'confianza': min(0.6, area / 3000),
                            'distancia_estimada': self.estimar_distancia_mejorada(w, h),
                            'detector': 'movimiento'
                        })
            
            self.frame_anterior = gray
            return resultados
        except Exception as e:
            print(f"❌ Error en movimiento: {e}")
            return []

    def estimar_distancia_mejorada(self, ancho_persona, alto_persona):
        """Estimación de distancia mejorada"""
        cal = self.calibracion_distancia
        
        dimension_promedio = (ancho_persona + alto_persona * 0.6) / 1.6
        
        if dimension_promedio <= 0:
            return 10.0
        
        if dimension_promedio >= cal['ancho_referencia_cerca']:
            ratio = cal['ancho_referencia_cerca'] / dimension_promedio
            distancia = cal['distancia_referencia_cerca'] * ratio
        elif dimension_promedio <= cal['ancho_referencia_lejos']:
            ratio = cal['ancho_referencia_lejos'] / dimension_promedio
            distancia = cal['distancia_referencia_lejos'] * ratio
        else:
            rango_ancho = cal['ancho_referencia_cerca'] - cal['ancho_referencia_lejos']
            rango_distancia = cal['distancia_referencia_lejos'] - cal['distancia_referencia_cerca']
            
            proporcion = (dimension_promedio - cal['ancho_referencia_lejos']) / rango_ancho
            distancia = cal['distancia_referencia_cerca'] + (rango_distancia * proporcion)
        
        distancia *= cal['factor_ajuste_camara']
        return max(0.2, min(15.0, distancia))

    def detectar_personas(self, frame):
        """Combina todas las detecciones"""
        todas_detecciones = []
        
        for nombre, detector in self.detectores:
            try:
                if nombre == 'movimiento':
                    detecciones = self.detectar_por_movimiento(frame)
                else:
                    detecciones = self.detectar_con_haar(frame, detector, nombre)
                
                todas_detecciones.extend(detecciones)
            except Exception as e:
                print(f"⚠️ Error en detector {nombre}: {e}")
        
        return self.filtrar_duplicados(todas_detecciones)

    def filtrar_duplicados(self, detecciones, umbral_solapamiento=0.5):
        """Filtra detecciones duplicadas"""
        if not detecciones:
            return []
            
        detecciones.sort(key=lambda x: x['confianza'], reverse=True)
        finales = []
        usadas = set()
        
        for i, det in enumerate(detecciones):
            if i in usadas:
                continue
                
            finales.append(det)
            x1_i, y1_i, x2_i, y2_i = det['bbox']
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            
            for j in range(i + 1, len(detecciones)):
                if j in usadas:
                    continue
                    
                x1_j, y1_j, x2_j, y2_j = detecciones[j]['bbox']
                x_left = max(x1_i, x1_j)
                y_top = max(y1_i, y1_j)
                x_right = min(x2_i, x2_j)
                y_bottom = min(y2_i, y2_j)
                
                if x_right < x_left or y_bottom < y_top:
                    continue
                    
                area_interseccion = (x_right - x_left) * (y_bottom - y_top)
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                
                solapamiento = area_interseccion / min(area_i, area_j)
                if solapamiento > umbral_solapamiento:
                    usadas.add(j)
        
        return finales

    def determinar_zona(self, centro_x):
        """Determina la zona de la persona"""
        for zona, (inicio, fin) in self.zonas.items():
            if inicio <= centro_x <= fin:
                return zona
        return 'frente'

    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja la interfaz en el frame (para mostrar en Streamlit)"""
        try:
            # Redimensionar si es necesario
            if frame.shape[1] != self.ancho_pantalla or frame.shape[0] != self.alto_pantalla:
                frame = cv2.resize(frame, (self.ancho_pantalla, self.alto_pantalla))
            
            # Líneas divisorias de zonas
            cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                    (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
            cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                    (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
            
            # Información de estado
            status_text = f"Personas: {len(detecciones)} | Modo: Cloud"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar detecciones
            for i, det in enumerate(detecciones):
                x1, y1, x2, y2 = det['bbox']
                distancia = det['distancia_estimada']
                zona = self.determinar_zona(det['centro'][0])
                
                # Color según distancia
                if distancia < 1.0:
                    color = (0, 0, 255)  # Rojo
                elif distancia < 2.0:
                    color = (0, 165, 255)  # Naranja
                else:
                    color = (0, 255, 0)  # Verde
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{i+1}: {distancia:.1f}m ({zona})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, det['centro'], 5, color, -1)
            
            return frame
        except Exception as e:
            print(f"❌ Error dibujando interfaz: {e}")
            return frame

    def procesar_imagen(self, image_file):
        """Procesa una imagen subida"""
        try:
            # Convertir imagen de Streamlit a formato OpenCV
            image = Image.open(image_file)
            frame = np.array(image)
            
            # Convertir RGB a BGR si es necesario
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detectar personas
            detecciones = self.detectar_personas(frame)
            
            # Dibujar interfaz
            frame_procesado = self.dibujar_interfaz(frame, detecciones)
            
            # Convertir de vuelta a RGB para Streamlit
            frame_rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)
            
            self.detecciones_actuales = detecciones
            return frame_rgb, detecciones
            
        except Exception as e:
            st.error(f"❌ Error procesando imagen: {e}")
            return None, []

    def procesar_video(self, video_file):
        """Procesa un video subido"""
        try:
            # Guardar video temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
            
            # Abrir video con OpenCV
            cap = cv2.VideoCapture(video_path)
            frames_procesados = []
            todas_detecciones = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                detecciones = self.detectar_personas(frame)
                frame_procesado = self.dibujar_interfaz(frame, detecciones)
                frame_rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)
                
                frames_procesados.append(frame_rgb)
                todas_detecciones.extend(detecciones)
            
            cap.release()
            os.unlink(video_path)
            
            self.detecciones_actuales = todas_detecciones
            return frames_procesados, todas_detecciones
            
        except Exception as e:
            st.error(f"❌ Error procesando video: {e}")
            return [], []

# Interfaz de Streamlit
def main():
    st.set_page_config(
        page_title="Sistema de Detección de Personas - Cloud",
        page_icon="👤",
        layout="wide"
    )
    
    # Inicializar sistema en session_state
    if 'sistema' not in st.session_state:
        st.session_state.sistema = SistemaDeteccionPersonasStreamlit()
        st.session_state.procesado = False
        st.session_state.resultados = None
    
    sistema = st.session_state.sistema
    
    st.title("👤 Sistema de Detección de Personas - STREAMLIT CLOUD")
    st.warning("""
    ☁️ **MODO STREAMLIT CLOUD ACTIVADO**
    - 📁 **Sube imágenes o videos** para procesar
    - 🔇 **Audio no disponible** en este entorno
    - ❌ **Cámara en vivo no disponible**
    - ✅ **Detección funciona** con archivos subidos
    """)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURACIÓN")
        
        st.subheader("📁 SUBIR ARCHIVO")
        tipo_archivo = st.radio(
            "Tipo de archivo:",
            ["Imagen", "Video"],
            horizontal=True
        )
        
        if tipo_archivo == "Imagen":
            archivo = st.file_uploader(
                "Sube una imagen",
                type=['jpg', 'jpeg', 'png'],
                help="Sube una imagen con personas para detectar"
            )
        else:
            archivo = st.file_uploader(
                "Sube un video", 
                type=['mp4', 'avi', 'mov'],
                help="Sube un video con personas para detectar"
            )
        
        st.subheader("🎯 CALIBRACIÓN")
        factor_actual = sistema.calibracion_distancia['factor_ajuste_camara']
        nuevo_factor = st.slider(
            "Factor de distancia:",
            0.5, 3.0, float(factor_actual), 0.1,
            help="Ajusta si las distancias no son precisas"
        )
        if nuevo_factor != factor_actual:
            sistema.calibracion_distancia['factor_ajuste_camara'] = nuevo_factor
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎬 PROCESAMIENTO")
        
        if archivo is not None:
            st.success(f"📁 **Archivo cargado:** {archivo.name}")
            
            if tipo_archivo == "Imagen":
                if st.button("🔍 PROCESAR IMAGEN", type="primary"):
                    with st.spinner("Procesando imagen..."):
                        frame_procesado, detecciones = sistema.procesar_imagen(archivo)
                    
                    if frame_procesado is not None:
                        st.image(frame_procesado, use_container_width=True, 
                                caption="Imagen procesada con detecciones")
                        
                        if detecciones:
                            st.success(f"✅ **{len(detecciones)} persona(s) detectada(s)**")
                        else:
                            st.info("❌ No se detectaron personas")
            
            else:  # Video
                if st.button("🎬 PROCESAR VIDEO", type="primary"):
                    with st.spinner("Procesando video..."):
                        frames, detecciones = sistema.procesar_video(archivo)
                    
                    if frames:
                        st.success(f"✅ **Video procesado:** {len(frames)} frames")
                        st.image(frames[-1], use_container_width=True,
                                caption="Último frame procesado")
                        
                        if detecciones:
                            st.info(f"📊 **Total de detecciones en video:** {len(detecciones)}")
                    else:
                        st.error("❌ Error al procesar el video")
        
        else:
            st.info("👆 **Selecciona y sube un archivo** para comenzar el procesamiento")
            st.image("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                    caption="Ejemplo - Tu imagen/video aparecerá aquí", 
                    use_container_width=True)
    
    with col2:
        st.subheader("📊 RESULTADOS")
        
        if sistema.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(sistema.detecciones_actuales)}")
            
            for i, det in enumerate(sistema.detecciones_actuales):
                with st.expander(f"Persona {i+1}", expanded=True):
                    st.metric("Distancia", f"{det['distancia_estimada']:.2f}m")
                    st.metric("Zona", sistema.determinar_zona(det['centro'][0]))
                    st.metric("Confianza", f"{det['confianza']:.0%}")
                    st.caption(f"Detector: {det['detector']}")
        else:
            st.info("📋 **Esperando datos...**")
            st.caption("Los resultados aparecerán aquí después del procesamiento")
        
        # Información de calibración
        st.markdown("---")
        st.subheader("⚙️ CONFIGURACIÓN ACTUAL")
        st.write(f"**Factor de calibración:** {sistema.calibracion_distancia['factor_ajuste_camara']:.1f}")
        st.write(f"**Detectores activos:** {len(sistema.detectores)}")

if __name__ == "__main__":
    main()
