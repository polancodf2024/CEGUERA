# deteccion_personas_video_streamlit.py
import cv2
import numpy as np
import time
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("MediaPipe no está instalado. Usando detección por movimiento.")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.sistema = SistemaDeteccionPersonasStreamlit()
        self.frame_count = 0
        
    def recv(self, frame):
        self.frame_count += 1
        
        # Convertir frame de PyAV a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Procesar cada frame para mejor detección
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

class SistemaDeteccionPersonasStreamlit:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.ultima_alerta = 0
        self.cooldown = 2.0
        
        print("✅ Sistema inicializado - Usando MediaPipe para detección")
        
    def inicializar_deteccion_avanzada(self):
        """Inicializa MediaPipe para detección de personas"""
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector_type = "mediapipe"
                print("✅ MediaPipe cargado correctamente")
            except Exception as e:
                print(f"❌ Error cargando MediaPipe: {e}")
                self.detector_type = "movimiento"
                self.frame_anterior = None
        else:
            # Fallback a detección por movimiento
            self.detector_type = "movimiento"
            self.frame_anterior = None
            print("✅ Usando detección por movimiento")
        
    def configurar_parametros(self):
        """Configura parámetros del sistema"""
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        # Zonas de la pantalla
        self.zonas = {
            'izquierda': (0, self.ancho_pantalla * 0.4),
            'frente': (self.ancho_pantalla * 0.4, self.ancho_pantalla * 0.6),
            'derecha': (self.ancho_pantalla * 0.6, self.ancho_pantalla)
        }
    
    def detectar_con_mediapipe(self, frame):
        """Detección usando MediaPipe Pose"""
        try:
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.pose.process(rgb_frame)
            
            if not resultados.pose_landmarks:
                return []
            
            # Obtener todos los landmarks
            landmarks = resultados.pose_landmarks.landmark
            
            # Filtrar landmarks visibles
            x_coords = [lm.x for lm in landmarks if lm.visibility > 0.3]
            y_coords = [lm.y for lm in landmarks if lm.visibility > 0.3]
            
            if len(x_coords) < 5 or len(y_coords) < 5:  # Mínimo de landmarks visibles
                return []
            
            # Calcular bounding box
            x_min = int(min(x_coords) * frame.shape[1])
            x_max = int(max(x_coords) * frame.shape[1])
            y_min = int(min(y_coords) * frame.shape[0])
            y_max = int(max(y_coords) * frame.shape[0])
            
            # Asegurar que la bounding box sea válida
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            
            ancho = x_max - x_min
            alto = y_max - y_min
            
            if ancho <= 0 or alto <= 0:
                return []
                
            centro_x = x_min + ancho // 2
            
            # Estimación de distancia basada en el tamaño
            area = ancho * alto
            distancia = self.estimar_distancia(area)
            
            return [{
                'bbox': (x_min, y_min, x_max, y_max),
                'centro': (centro_x, y_min + alto // 2),
                'confianza': 0.8,
                'distancia_estimada': distancia,
                'detector': 'mediapipe'
            }]
            
        except Exception as e:
            print(f"❌ Error en MediaPipe: {e}")
            return []
    
    def detectar_por_movimiento(self, frame):
        """Detección por movimiento (fallback)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.frame_anterior is None:
                self.frame_anterior = gray
                return []
            
            frame_diff = cv2.absdiff(self.frame_anterior, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            resultados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 1000:  # Área mínima mayor para filtrar ruido
                    x, y, w, h = cv2.boundingRect(contorno)
                    
                    # Filtrar por relación de aspecto (personas son más altas que anchas)
                    relacion = h / w if w > 0 else 0
                    if 1.2 < relacion < 4.0:
                        centro_x = x + w // 2
                        distancia = self.estimar_distancia(w * h)
                        
                        resultados.append({
                            'bbox': (x, y, x + w, y + h),
                            'centro': (centro_x, y + h // 2),
                            'confianza': min(0.7, area / 5000),
                            'distancia_estimada': distancia,
                            'detector': 'movimiento'
                        })
            
            self.frame_anterior = gray
            return resultados
        except Exception as e:
            print(f"❌ Error en movimiento: {e}")
            return []

    def estimar_distancia(self, area):
        """Estimación simple de distancia basada en área"""
        if area <= 0:
            return 10.0
        
        # Calibración empírica - ajustar según tu cámara
        if area > 50000:
            return 0.5
        elif area > 30000:
            return 1.0
        elif area > 15000:
            return 2.0
        elif area > 8000:
            return 3.0
        elif area > 4000:
            return 4.0
        else:
            return 5.0
    
    def detectar_personas(self, frame):
        """Combina detecciones de todos los métodos"""
        try:
            if self.detector_type == "mediapipe" and MEDIAPIPE_AVAILABLE:
                return self.detectar_con_mediapipe(frame)
            else:
                return self.detectar_por_movimiento(frame)
        except Exception as e:
            print(f"❌ Error en detección principal: {e}")
            return []
    
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
        
        if distancia < 1.5:
            mensaje = f"🚨 ALERTA! Persona CERCA a {distancia:.1f}m - {zona.upper()}"
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
        
        print(f"👤 {mensaje} | Detector: {persona_cercana['detector']}")
        self.ultima_alerta = tiempo_actual
    
    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja interfaz con detecciones"""
        try:
            # Redimensionar si es necesario
            if frame.shape[1] != self.ancho_pantalla or frame.shape[0] != self.alto_pantalla:
                frame = cv2.resize(frame, (self.ancho_pantalla, self.alto_pantalla))
            
            # Dibujar zonas
            cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                    (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
            cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                    (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
            
            # Etiquetar zonas
            cv2.putText(frame, "IZQUIERDA", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "FRENTE", (int(self.ancho_pantalla * 0.45), 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "DERECHA", (int(self.ancho_pantalla * 0.65), 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dibujar landmarks de MediaPipe si está disponible y hay detecciones
            if MEDIAPIPE_AVAILABLE and self.detector_type == "mediapipe":
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resultados = self.pose.process(rgb_frame)
                    if resultados.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, 
                            resultados.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                except Exception as e:
                    print(f"⚠️ Error dibujando landmarks: {e}")
            
            # Dibujar detecciones
            for i, det in enumerate(detecciones):
                x1, y1, x2, y2 = det['bbox']
                distancia = det['distancia_estimada']
                zona = self.determinar_zona(det['centro'][0])
                
                if distancia < 1.5:
                    color = (0, 0, 255)  # Rojo para cercano
                elif distancia < 3.0:
                    color = (0, 165, 255)  # Naranja
                else:
                    color = (0, 255, 0)  # Verde para lejano
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Etiqueta de información
                label = f"Persona {i+1}: {distancia:.1f}m ({zona})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Punto central
                cv2.circle(frame, det['centro'], 8, color, -1)
                
                # Información adicional
                info_text = f"Conf: {det['confianza']:.0%} | {det['detector']}"
                cv2.putText(frame, info_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Estado del sistema
            status_color = (0, 255, 0) if detecciones else (0, 0, 255)
            status_text = f"Personas: {len(detecciones)} | {self.detector_type.upper()}"
            cv2.putText(frame, status_text, (10, self.alto_pantalla - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
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
    
    # Información del sistema
    if MEDIAPIPE_AVAILABLE:
        st.success("**✅ MEDIAPIPE ACTIVADO** - Detección avanzada de personas")
    else:
        st.warning("**⚠️ DETECCIÓN POR MOVIMIENTO** - Instala MediaPipe para mejor precisión")
        st.code("pip install mediapipe")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURACIÓN")
        
        st.subheader("🎯 INFORMACIÓN DEL SISTEMA")
        if MEDIAPIPE_AVAILABLE:
            st.success("MediaPipe ✅ Disponible")
            detector_info = "MediaPipe Pose"
        else:
            st.warning("MediaPipe ❌ No disponible")
            detector_info = "Detección por Movimiento"
        
        st.metric("Detector Actual", detector_info)
        
        st.subheader("📊 ESTADÍSTICAS EN VIVO")
        if st.session_state.alertas:
            total_alertas = len(st.session_state.alertas)
            alertas_cercanas = len([a for a in st.session_state.alertas if a['distancia'] < 1.5])
            
            st.metric("Alertas Totales", total_alertas)
            st.metric("Alertas Cercanas", alertas_cercanas)
            
            # Persona más cercana actual
            if st.session_state.detecciones_actuales:
                persona_cercana = min(st.session_state.detecciones_actuales, 
                                    key=lambda x: x['distancia_estimada'])
                st.metric("Distancia Mínima", f"{persona_cercana['distancia_estimada']:.1f}m")
                st.metric("Zona Actual", persona_cercana.get('zona', 'N/A'))
            else:
                st.info("Esperando detecciones...")
        else:
            st.info("Esperando detecciones...")
        
        st.subheader("🔧 AJUSTES")
        mostrar_landmarks = st.checkbox("Mostrar puntos corporales", value=True)
        modo_debug = st.checkbox("Modo depuración", value=False)
        
        st.subheader("💡 CONSEJOS")
        st.info("""
        - **Buena iluminación** mejora la detección
        - **Persona completa** en el frame
        - **Evitar movimientos bruscos**
        - **Distancia óptima**: 1-3 metros
        """)
    
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
                    "height": {"min": 480, "ideal": 720},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        if not webrtc_ctx.state.playing:
            st.info("🎥 **Haz clic en 'START' para activar la cámara**")
            st.image("https://via.placeholder.com/640x480/2E86AB/FFFFFF?text=VIDEO+EN+VIVO", 
                    caption="Vista previa - El video en vivo aparecerá aquí", 
                    use_container_width=True)
            
            # Mostrar información de configuración
            with st.expander("🔧 Configuración de la cámara"):
                st.write("""
                **Resolución recomendada:** 640x480 o 1280x720
                **Frame rate:** 30 FPS
                **Formato:** RGB
                
                **Si la cámara no funciona:**
                1. Verifica los permisos del navegador
                2. Asegúrate de que solo una app use la cámara
                3. Prueba en un entorno bien iluminado
                """)
        else:
            st.success("✅ **Cámara activa** - Detectando personas en tiempo real...")
            
            # Mostrar información del stream
            if st.session_state.detecciones_actuales:
                st.balloon()  # Efecto visual cuando detecta
    
    with col2:
        st.subheader("📊 DETECCIONES EN VIVO")
        
        if st.session_state.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(st.session_state.detecciones_actuales)}")
            
            for i, det in enumerate(st.session_state.detecciones_actuales):
                with st.expander(f"Persona {i+1}", expanded=True):
                    distancia = det['distancia_estimada']
                    zona = det.get('zona', 'frente')
                    
                    # Indicador de distancia
                    if distancia < 1.5:
                        st.error(f"🚨 **MUY CERCA:** {distancia:.1f}m")
                    elif distancia < 3.0:
                        st.warning(f"⚠️ **CERCA:** {distancia:.1f}m")
                    else:
                        st.success(f"✅ **SEGURO:** {distancia:.1f}m")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Zona", zona)
                    with col_b:
                        st.metric("Confianza", f"{det['confianza']:.0%}")
                    
                    # Información adicional
                    ancho = det['bbox'][2] - det['bbox'][0]
                    alto = det['bbox'][3] - det['bbox'][1]
                    st.caption(f"**Tamaño:** {ancho}×{alto}px")
                    st.caption(f"**Detector:** {det['detector']}")
        else:
            st.info("👀 **Monitoreando...**")
            st.caption("Las personas detectadas aparecerán aquí automáticamente")
            
            # Placeholder para demostración
            with st.expander("🔍 Ejemplo de detección"):
                st.metric("Distancia", "2.5m")
                st.metric("Zona", "frente")
                st.metric("Confianza", "85%")
        
        # Alertas en tiempo real
        if st.session_state.alertas:
            st.subheader("🚨 ALERTAS RECIENTES")
            for alerta in reversed(st.session_state.alertas[-5:]):
                tiempo = time.strftime('%H:%M:%S', time.localtime(alerta['timestamp']))
                if alerta['distancia'] < 1.5:
                    st.error(f"**{tiempo}** - {alerta['mensaje']}")
                else:
                    st.warning(f"**{tiempo}** - {alerta['mensaje']}")
    
    # Información adicional
    st.markdown("---")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.subheader("🎯 ZONAS DE DETECCIÓN")
        st.write("""
        - **🔴 IZQUIERDA:** 0-40% del frame
        - **🟢 FRENTE:** 40-60% del frame  
        - **🔵 DERECHA:** 60-100% del frame
        """)
    
    with col_info2:
        st.subheader("📏 INDICADORES DE DISTANCIA")
        st.write("""
        - **🔴 ROJO:** < 1.5m (ALERTA)
        - **🟠 NARANJA:** 1.5-3m (PRECAUCIÓN)
        - **🟢 VERDE:** > 3m (SEGURO)
        """)
    
    with col_info3:
        st.subheader("🔧 DETECTORES")
        st.write("""
        - **MediaPipe:** Puntos corporales (Recomendado)
        - **Movimiento:** Análisis de diferencia entre frames
        - **Haar Cascade:** Detección por características (Legacy)
        """)
    
    st.markdown("---")
    st.info("""
    **💡 Instrucciones de uso:**
    1. Haz clic en **START** para activar la cámara
    2. Permite el acceso a la cámara cuando el navegador lo solicite
    3. Apunta la cámara hacia el área que quieres monitorear
    4. Las detecciones y alertas aparecerán automáticamente
    5. Haz clic en **STOP** para desactivar la cámara
    
    **🎯 Para mejor detección:**
    - Buena iluminación en el área
    - Personas completas en el frame
    - Distancia de 1-4 metros de la cámara
    - Movimientos suaves y naturales
    """)

if __name__ == "__main__":
    main()
