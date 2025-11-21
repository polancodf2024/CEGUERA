# deteccion_personas_video_streamlit.py
import cv2
import numpy as np
import time
import pygame
import os
import streamlit as st
from PIL import Image
import tempfile
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.sistema = SistemaDeteccionPersonasStreamlit()
        self.frame_count = 0
        
    def recv(self, frame):
        self.frame_count += 1
        
        # Convertir frame de PyAV a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Procesar cada 2 frames para mejor rendimiento
        if self.frame_count % 2 == 0:
            detecciones = self.sistema.detectar_personas(img)
            
            # Generar alertas si hay detecciones
            if detecciones:
                self.sistema.generar_alerta_automatica(detecciones)
            
            # Dibujar interfaz en el frame
            img_procesado = self.sistema.dibujar_interfaz(img, detecciones)
            
            # Actualizar detecciones en session state
            if 'detecciones_actuales' not in st.session_state:
                st.session_state.detecciones_actuales = []
            st.session_state.detecciones_actuales = detecciones
            
            return av.VideoFrame.from_ndarray(img_procesado, format="bgr24")
        
        return frame

class SistemaDeteccionPersonasStreamlit:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.frame_actual = None
        self.ultima_alerta = 0
        self.cooldown = 2.0
        
        # Inicializar componentes
        self.audio_disponible = True
        self.sonidos = {}
        self.crear_sonidos_simples()
        
        print("✅ Sistema inicializado - Modo Video en Tiempo Real")
        
    def crear_sonidos_simples(self):
        """Sonidos simples para notificaciones"""
        self.sonidos = {
            'izquierda': 'left',
            'derecha': 'right', 
            'frente': 'center',
            'alerta': 'alert'
        }
    
    def reproducir_alerta(self, zona, distancia):
        """Reproduce alerta según la zona y distancia"""
        if not self.audio_disponible:
            return
            
        try:
            if distancia < 1.0:
                # Usar markdown para alertas más visibles
                st.markdown(f"""
                <div style='background-color: #ff4444; padding: 10px; border-radius: 5px; color: white;'>
                🚨 **ALERTA!** Persona MUY CERCA a {distancia:.1f}m - {zona.upper()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #44ff44; padding: 10px; border-radius: 5px; color: black;'>
                🔊 Persona detectada: {distancia:.1f}m - {zona}
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            print(f"❌ Error en alerta: {e}")
    
    def inicializar_deteccion_avanzada(self):
        """Inicializa detectores mejorados"""
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
        
        # Haar Cascade para cuerpo superior
        try:
            haar_upper_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            if os.path.exists(haar_upper_path):
                detector_upper = cv2.CascadeClassifier(haar_upper_path)
                if not detector_upper.empty():
                    self.detectores.append(('haar_upperbody', detector_upper))
                    print("✅ Haar Upperbody cargado")
        except Exception as e:
            print(f"⚠️ Error Haar Upperbody: {e}")
            
        # Detección por movimiento
        self.frame_anterior = None
        self.detectores.append(('movimiento', None))
        print("✅ Detección por movimiento disponible")
        
        if not self.detectores:
            print("❌ No hay detectores disponibles")
    
    def configurar_parametros(self):
        """Configura parámetros del sistema"""
        self.umbral_confianza = 0.3
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        # Parámetros de calibración
        self.calibracion_distancia = {
            'ancho_referencia_cerca': 200,
            'ancho_referencia_lejos': 50,
            'distancia_referencia_cerca': 0.4,
            'distancia_referencia_lejos': 2.0,
            'factor_ajuste_camara': 1.0
        }
        
        # Zonas de la pantalla
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
                min_size = (20, 40)
                scale_factor = 1.01
                min_neighbors = 2
            else:
                min_size = (20, 20)
                scale_factor = 1.01
                min_neighbors = 2
                
            detecciones = detector.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            resultados = []
            for (x, y, w, h) in detecciones:
                centro_x = x + w//2
                centro_y = y + h//2
                
                area = w * h
                confianza = min(0.9, area / 5000)
                
                resultados.append({
                    'bbox': (x, y, x + w, y + h),
                    'centro': (centro_x, centro_y),
                    'confianza': confianza,
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
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            
            if self.frame_anterior is None:
                self.frame_anterior = gray
                return []
            
            frame_diff = cv2.absdiff(self.frame_anterior, gray)
            thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            resultados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contorno)
                    
                    relacion = h / w if w > 0 else 0
                    if 1.0 < relacion < 5.0:
                        centro_x = x + w//2
                        
                        confianza = min(0.8, area / 2000)
                        
                        resultados.append({
                            'bbox': (x, y, x + w, y + h),
                            'centro': (centro_x, y + h//2),
                            'confianza': confianza,
                            'distancia_estimada': self.estimar_distancia_mejorada(w, h),
                            'detector': 'movimiento'
                        })
            
            self.frame_anterior = gray
            return resultados
        except Exception as e:
            print(f"❌ Error en movimiento: {e}")
            return []

    def estimar_distancia_mejorada(self, ancho_persona, alto_persona):
        """Estimación de distancia"""
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
        """Combina detecciones de todos los métodos"""
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
        
        return self.filtrar_duplicados(todas_detecciones, umbral_solapamiento=0.3)

    def filtrar_duplicados(self, detecciones, umbral_solapamiento=0.3):
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
        """Determina en qué zona está la persona"""
        for zona, (inicio, fin) in self.zonas.items():
            if inicio <= centro_x <= fin:
                return zona
        return 'frente'
    
    def generar_alerta_automatica(self, detecciones):
        """Genera alertas automáticamente cuando detecta personas"""
        tiempo_actual = time.time()
        
        if tiempo_actual - self.ultima_alerta < self.cooldown:
            return
        
        if not detecciones:
            return
        
        persona_cercana = min(detecciones, key=lambda x: x['distancia_estimada'])
        distancia = persona_cercana['distancia_estimada']
        zona = self.determinar_zona(persona_cercana['centro'][0])
        
        if 'alertas' not in st.session_state:
            st.session_state.alertas = []
        
        if distancia < 1.0:
            mensaje = f"🚨 ALERTA! Persona MUY CERCA a {distancia:.1f}m - {zona.upper()}"
        else:
            mensaje = f"🔊 Persona detectada: {distancia:.1f}m - {zona}"
        
        st.session_state.alertas.append({
            'timestamp': time.time(),
            'mensaje': mensaje,
            'distancia': distancia,
            'zona': zona
        })
        
        if len(st.session_state.alertas) > 10:
            st.session_state.alertas.pop(0)
        
        # Solo mostrar alerta si estamos en el contexto principal
        if not hasattr(self, 'en_video_processor'):
            self.reproducir_alerta(zona, distancia)
        
        print(f"👤 {mensaje} | Detector: {persona_cercana['detector']}")
        self.ultima_alerta = tiempo_actual
    
    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja interfaz con detecciones"""
        try:
            if frame.shape[1] != self.ancho_pantalla or frame.shape[0] != self.alto_pantalla:
                frame = cv2.resize(frame, (self.ancho_pantalla, self.alto_pantalla))
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.ancho_pantalla, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                    (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
            cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                    (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
            
            status_color = (0, 255, 0) if detecciones else (0, 0, 255)
            status_text = f"Personas: {len(detecciones)} | VIDEO EN VIVO"
            cv2.putText(frame, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            for i, det in enumerate(detecciones):
                x1, y1, x2, y2 = det['bbox']
                distancia = det['distancia_estimada']
                zona = self.determinar_zona(det['centro'][0])
                
                if distancia < 1.0:
                    color = (0, 0, 255)
                elif distancia < 2.0:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label = f"Persona {i+1}: {distancia:.1f}m ({zona})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.circle(frame, det['centro'], 8, color, -1)
                
                info_text = f"Conf: {det['confianza']:.0%}"
                cv2.putText(frame, info_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return frame
        except Exception as e:
            print(f"❌ Error dibujando interfaz: {e}")
            return frame

# INTERFAZ PRINCIPAL CON VIDEO EN VIVO
def main():
    st.set_page_config(
        page_title="Detección de Personas - Video en Vivo",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar session state
    if 'alertas' not in st.session_state:
        st.session_state.alertas = []
    if 'detecciones_actuales' not in st.session_state:
        st.session_state.detecciones_actuales = []
    
    st.title("🎥 Sistema de Detección de Personas - VIDEO EN VIVO")
    st.success("**VIDEO EN TIEMPO REAL ACTIVADO** - La detección funciona sobre el video en vivo")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURACIÓN")
        
        st.subheader("🎯 SENSIBILIDAD")
        sensibilidad = st.slider(
            "Nivel de detección:",
            1, 10, 7,
            help="Ajusta qué tan sensible es la detección"
        )
        
        st.subheader("🔊 ALERTAS")
        alertas_activas = st.checkbox("Activar alertas de audio", value=True)
        
        st.subheader("📊 ESTADÍSTICAS EN VIVO")
        if st.session_state.alertas:
            total_alertas = len(st.session_state.alertas)
            alertas_cercanas = len([a for a in st.session_state.alertas if a['distancia'] < 1.0])
            
            st.metric("Alertas Totales", total_alertas)
            st.metric("Alertas Cercanas", alertas_cercanas)
            
            # Persona más cercana actual
            if st.session_state.detecciones_actuales:
                persona_cercana = min(st.session_state.detecciones_actuales, 
                                    key=lambda x: x['distancia_estimada'])
                st.metric("Distancia Mínima", f"{persona_cercana['distancia_estimada']:.1f}m")
        else:
            st.info("Esperando detecciones...")
    
    # Área principal - VIDEO EN VIVO
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 VIDEO EN VIVO")
        
        # Configuración para WebRTC
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Widget de video en vivo con WebRTC
        webrtc_ctx = webrtc_streamer(
            key="video-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280},
                    "height": {"min": 480, "ideal": 720}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        if not webrtc_ctx.state.playing:
            st.info("🎥 **Haz clic en 'START' para activar la cámara**")
            st.image("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                    caption="Vista previa - El video en vivo aparecerá aquí", 
                    use_container_width=True)
        else:
            st.success("✅ **Cámara activa** - Detectando personas en tiempo real...")
    
    with col2:
        st.subheader("📊 DETECCIONES EN VIVO")
        
        if st.session_state.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(st.session_state.detecciones_actuales)}")
            
            for i, det in enumerate(st.session_state.detecciones_actuales):
                with st.expander(f"Persona {i+1}", expanded=True):
                    distancia = det['distancia_estimada']
                    zona = det.get('zona', 'frente')
                    
                    st.metric("Distancia", f"{distancia:.2f}m")
                    st.metric("Zona", zona)
                    st.metric("Confianza", f"{det['confianza']:.0%}")
                    
                    ancho = det['bbox'][2] - det['bbox'][0]
                    alto = det['bbox'][3] - det['bbox'][1]
                    st.caption(f"Tamaño: {ancho}×{alto}px")
                    st.caption(f"Detector: {det['detector']}")
        else:
            st.info("👀 **Monitoreando...**")
            st.caption("Las personas detectadas aparecerán aquí automáticamente")
        
        # Alertas en tiempo real
        if st.session_state.alertas:
            st.subheader("🚨 ALERTAS RECIENTES")
            for alerta in st.session_state.alertas[-5:]:
                tiempo = time.strftime('%H:%M:%S', time.localtime(alerta['timestamp']))
                if alerta['distancia'] < 1.0:
                    st.error(f"**{tiempo}** - {alerta['mensaje']}")
                else:
                    st.warning(f"**{tiempo}** - {alerta['mensaje']}")
    
    # Información adicional
    st.markdown("---")
    st.info("""
    **💡 Instrucciones:**
    1. Haz clic en **START** para activar la cámara
    2. Permite el acceso a la cámara cuando el navegador lo solicite
    3. Apunta la cámara hacia el área que quieres monitorear
    4. Las detecciones y alertas aparecerán automáticamente
    5. Haz clic en **STOP** para desactivar la cámara
    """)

if __name__ == "__main__":
    main()
