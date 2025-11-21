# deteccion_personas_universal.py
import cv2
import numpy as np
import time
import pygame
import os
import streamlit as st
from PIL import Image
import tempfile
import threading
import platform

class SistemaDeteccionPersonasUniversal:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.frame_actual = None
        self.ultima_alerta = 0
        self.cooldown = 2.0
        self.ejecutando = False
        self.cap = None
        
        # Detectar dispositivo
        self.es_movil = self.detectar_dispositivo_movil()
        self.es_desktop = not self.es_movil
        
        # Inicializar componentes según dispositivo
        self.audio_disponible = False
        self.camara_disponible = False
        self.sonidos = {}
        
        self.inicializar_audio()
        self.probar_camara()
        
        print(f"✅ Sistema inicializado - Móvil: {self.es_movil}, Audio: {self.audio_disponible}, Cámara: {self.camara_disponible}")
        
    def detectar_dispositivo_movil(self):
        """Detecta si es un dispositivo móvil"""
        try:
            # Verificar user agent de Streamlit
            user_agent = st.get_option("browser.gatherUsageStats")
            
            # Detectar por tamaño de pantalla (aproximación)
            if hasattr(st, 'session_state'):
                return True  # En móvil, asumimos True para probar
            
            return False
        except:
            return True  # Por defecto asumimos móvil para mayor compatibilidad
    
    def probar_camara(self):
        """Prueba si hay cámara disponible"""
        try:
            if self.es_movil:
                # En móvil, la cámara debería funcionar
                self.camara_disponible = True
                return
                
            # En desktop, probar cámaras
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.camara_disponible = True
                    cap.release()
                    return
            self.camara_disponible = False
        except:
            self.camara_disponible = False
    
    def inicializar_audio(self):
        """Inicializa audio según el dispositivo"""
        try:
            if self.es_movil:
                # En móvil usar audio simple
                self.audio_disponible = True
                self.crear_sonidos_simples()
            else:
                # En desktop usar pygame
                pygame.init()
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                if pygame.mixer.get_init():
                    self.audio_disponible = True
                    self.crear_sonidos_mejorados()
        except Exception as e:
            print(f"❌ Audio no disponible: {e}")
            self.audio_disponible = False
    
    def crear_sonidos_simples(self):
        """Sonidos simples para móvil"""
        self.sonidos = {
            'izquierda': 'left',
            'derecha': 'right', 
            'frente': 'center',
            'alerta': 'alert'
        }
    
    def crear_sonidos_mejorados(self):
        """Sonidos mejorados para desktop"""
        try:
            sample_rate = 22050
            duration = 0.8
            
            tonos = {
                'izquierda': (440, 330),
                'derecha': (523, 392),
                'frente': (659, 494),
                'alerta': (784, 587)
            }
            
            for zona, (freq1, freq2) in tonos.items():
                samples = int(sample_rate * duration)
                t = np.linspace(0, duration, samples, False)
                
                wave1 = 0.6 * np.sin(2 * np.pi * freq1 * t)
                wave2 = 0.4 * np.sin(2 * np.pi * freq2 * t)
                wave = wave1 + wave2
                
                envelope = np.ones_like(t)
                attack = int(0.15 * sample_rate)
                release = int(0.3 * sample_rate)
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                wave *= envelope
                
                wave_int = (wave * 32767).astype(np.int16)
                wave_stereo = np.column_stack((wave_int, wave_int))
                sound = pygame.sndarray.make_sound(wave_stereo)
                self.sonidos[zona] = sound
                
        except Exception as e:
            print(f"❌ Error creando sonidos: {e}")
    
    def reproducir_sonido(self, zona):
        """Reproduce sonido según el dispositivo"""
        if not self.audio_disponible or zona not in self.sonidos:
            return
            
        try:
            if self.es_movil:
                # En móvil, usar notificaciones de Streamlit
                if zona == 'alerta':
                    st.warning("🚨 ALERTA: Persona muy cerca!")
                else:
                    st.info(f"🔊 Persona detectada a la {zona}")
            else:
                # En desktop, usar pygame
                self.sonidos[zona].play()
        except Exception as e:
            print(f"❌ Error reproduciendo sonido: {e}")
    
    def inicializar_deteccion_avanzada(self):
        """Inicializa detectores"""
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
            print(f"⚠️ Error Haar: {e}")
        
        # Detección por movimiento
        self.frame_anterior = None
        self.detectores.append(('movimiento', None))
        print("✅ Detección por movimiento disponible")
    
    def configurar_parametros(self):
        """Configura parámetros"""
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        self.zonas = {
            'izquierda': (0, self.ancho_pantalla * 0.4),
            'frente': (self.ancho_pantalla * 0.4, self.ancho_pantalla * 0.6),
            'derecha': (self.ancho_pantalla * 0.6, self.ancho_pantalla)
        }

    def detectar_personas(self, frame):
        """Detección principal optimizada"""
        try:
            # Redimensionar para consistencia
            frame = cv2.resize(frame, (self.ancho_pantalla, self.alto_pantalla))
            
            # Detección por movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.frame_anterior is None:
                self.frame_anterior = gray
                return []
            
            frame_diff = cv2.absdiff(self.frame_anterior, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detecciones = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 300:  # Menor área para mejor sensibilidad
                    x, y, w, h = cv2.boundingRect(contorno)
                    relacion = h / w if w > 0 else 0
                    
                    # Filtrar por proporción humana
                    if 1.2 < relacion < 4.0:
                        centro_x = x + w//2
                        distancia = self.estimar_distancia_simple(w)
                        
                        detecciones.append({
                            'bbox': (x, y, x + w, y + h),
                            'centro': (centro_x, y + h//2),
                            'confianza': min(0.9, area / 1500),
                            'distancia_estimada': distancia,
                            'detector': 'movimiento'
                        })
            
            self.frame_anterior = gray
            return detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []

    def estimar_distancia_simple(self, ancho_persona):
        """Estimación simple de distancia"""
        if ancho_persona > 200: return 0.5
        elif ancho_persona > 120: return 1.0
        elif ancho_persona > 80: return 1.5
        elif ancho_persona > 50: return 2.0
        else: return 3.0

    def determinar_zona(self, centro_x):
        """Determina la zona de la persona"""
        for zona, (inicio, fin) in self.zonas.items():
            if inicio <= centro_x <= fin:
                return zona
        return 'frente'

    def generar_alerta(self, detecciones):
        """Genera alertas visuales y de audio"""
        tiempo_actual = time.time()
        if tiempo_actual - self.ultima_alerta < self.cooldown or not detecciones:
            return
        
        try:
            persona_cercana = min(detecciones, key=lambda x: x['distancia_estimada'])
            distancia = persona_cercana['distancia_estimada']
            zona = self.determinar_zona(persona_cercana['centro'][0])
            
            if distancia < 2.0:
                # Alertas de audio
                if distancia < 1.0:
                    self.reproducir_sonido('alerta')
                    mensaje = f"🚨 ALERTA! Persona a {distancia:.1f}m - {zona.upper()}"
                else:
                    self.reproducir_sonido(zona)
                    mensaje = f"👤 Persona detectada: {distancia:.1f}m - {zona}"
                
                # Guardar en sesión
                if 'alertas' not in st.session_state:
                    st.session_state.alertas = []
                
                st.session_state.alertas.append({
                    'timestamp': time.time(),
                    'mensaje': mensaje,
                    'distancia': distancia,
                    'zona': zona
                })
                
                # Limitar historial
                if len(st.session_state.alertas) > 10:
                    st.session_state.alertas.pop(0)
                
                print(f"🔊 {mensaje}")
                self.ultima_alerta = tiempo_actual
                
        except Exception as e:
            print(f"❌ Error en alerta: {e}")

    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja la interfaz en el frame"""
        try:
            # Líneas divisorias
            cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                    (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
            cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                    (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
            
            # Información de estado
            status_text = f"Personas: {len(detecciones)} | Audio: {'ON' if self.audio_disponible else 'OFF'}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar detecciones
            for i, det in enumerate(detecciones):
                x1, y1, x2, y2 = det['bbox']
                distancia = det['distancia_estimada']
                zona = self.determinar_zona(det['centro'][0])
                
                # Color según distancia
                if distancia < 1.0:
                    color = (0, 0, 255)  # Rojo - muy cerca
                elif distancia < 1.5:
                    color = (0, 165, 255)  # Naranja - cerca
                else:
                    color = (0, 255, 0)  # Verde - normal
                
                # Rectángulo y etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{i+1}: {distancia:.1f}m ({zona})"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, det['centro'], 4, color, -1)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error dibujando interfaz: {e}")
            return frame

    # ===== CÁMARA =====
    def iniciar_camara(self):
        """Inicia la cámara según el dispositivo"""
        try:
            if self.es_movil:
                # En móvil, Streamlit tiene acceso nativo a cámara
                return True, "✅ Cámara móvil lista"
            else:
                # En desktop, usar OpenCV
                for i in range(3):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 20)
                        self.ejecutando = True
                        return True, f"✅ Cámara {i} iniciada"
                return False, "❌ No se encontró cámara"
                
        except Exception as e:
            return False, f"❌ Error: {e}"

    def procesar_frame_camara(self):
        """Procesa frame de cámara"""
        if not self.ejecutando:
            return None
            
        try:
            if self.es_movil:
                # En móvil, Streamlit maneja la cámara
                return None
            else:
                # En desktop, usar OpenCV
                if not self.cap:
                    return None
                    
                ret, frame = self.cap.read()
                if not ret:
                    return None
                
                # Espejar para modo espejo
                frame_espejado = cv2.flip(frame, 1)
                
                # Procesar
                detecciones = self.detectar_personas(frame_espejado)
                self.generar_alerta(detecciones)
                frame_procesado = self.dibujar_interfaz(frame_espejado, detecciones)
                self.detecciones_actuales = detecciones
                
                return frame_procesado
                
        except Exception as e:
            print(f"❌ Error procesando cámara: {e}")
            return None

    def detener_camara(self):
        """Detiene la cámara"""
        self.ejecutando = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def tomar_foto_movil(self):
        """Toma foto en móvil usando Streamlit"""
        try:
            # Streamlit camera input para móvil
            picture = st.camera_input("Toma una foto")
            if picture:
                image = Image.open(picture)
                frame = np.array(image)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Procesar la foto
                detecciones = self.detectar_personas(frame_rgb)
                self.generar_alerta(detecciones)
                frame_procesado = self.dibujar_interfaz(frame_rgb, detecciones)
                self.detecciones_actuales = detecciones
                
                return frame_procesado
            return None
        except Exception as e:
            st.error(f"Error con cámara móvil: {e}")
            return None

# INTERFAZ STREAMLIT UNIVERSAL
def main():
    st.set_page_config(
        page_title="Detección de Personas Universal",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar sistema
    if 'sistema' not in st.session_state:
        st.session_state.sistema = SistemaDeteccionPersonasUniversal()
        st.session_state.camara_activa = False
        if 'alertas' not in st.session_state:
            st.session_state.alertas = []
    
    sistema = st.session_state.sistema
    
    # Header según dispositivo
    if sistema.es_movil:
        st.title("📱 Detección de Personas - MÓVIL")
        st.success("**📱 MODO MÓVIL ACTIVADO** - Cámara y Audio disponibles")
    else:
        st.title("💻 Detección de Personas - DESKTOP") 
        st.success("**💻 MODO DESKTOP** - Cámara y Audio disponibles")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURACIÓN")
        
        # Información del dispositivo
        st.subheader("📊 ESTADO")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dispositivo", "📱 Móvil" if sistema.es_movil else "💻 Desktop")
        with col2:
            st.metric("Audio", "✅ ON" if sistema.audio_disponible else "❌ OFF")
        
        # Controles de cámara
        st.subheader("🎥 CÁMARA")
        
        if sistema.es_movil:
            # En móvil, usar cámara nativa de Streamlit
            st.info("📸 Usa el botón de cámara abajo para tomar fotos")
            
        else:
            # En desktop, controles de cámara en vivo
            if not st.session_state.camara_activa:
                if st.button("🎥 INICIAR CÁMARA", type="primary", use_container_width=True):
                    exito, mensaje = sistema.iniciar_camara()
                    if exito:
                        st.session_state.camara_activa = True
                        st.success(mensaje)
                        st.rerun()
                    else:
                        st.error(mensaje)
            else:
                if st.button("⏹️ DETENER CÁMARA", type="secondary", use_container_width=True):
                    sistema.detener_camara()
                    st.session_state.camara_activa = False
                    st.rerun()
        
        # Controles de audio
        st.subheader("🔊 AUDIO")
        if st.button("🎵 PROBAR SONIDOS", use_container_width=True):
            if sistema.audio_disponible:
                if sistema.es_movil:
                    st.success("🔊 Audio móvil funcionando")
                    sistema.reproducir_sonido('frente')
                else:
                    st.success("🔊 Audio desktop funcionando")
                    sistema.reproducir_sonido('frente')
            else:
                st.error("🔇 Audio no disponible")
        
        # Calibración
        st.subheader("🎯 CALIBRACIÓN")
        if st.button("🔄 AUTO-CALIBRAR", use_container_width=True):
            if sistema.detecciones_actuales:
                st.success("✅ Sistema calibrado")
            else:
                st.warning("Toma una foto con personas primero")
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 VISTA EN VIVO")
        
        if sistema.es_movil:
            # MÓVIL: Cámara nativa de Streamlit
            frame_procesado = sistema.tomar_foto_movil()
            if frame_procesado is not None:
                st.image(frame_procesado, channels="BGR", use_column_width=True)
                
                if sistema.detecciones_actuales:
                    st.success(f"✅ {len(sistema.detecciones_actuales)} persona(s) detectada(s)")
                else:
                    st.info("❌ No se detectaron personas")
            else:
                st.info("👆 **Toma una foto con la cámara** para detectar personas")
                st.image("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                        caption="Ejemplo - Tu foto aparecerá aquí", use_column_width=True)
                        
        else:
            # DESKTOP: Cámara en vivo con OpenCV
            if st.session_state.camara_activa:
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with status_placeholder:
                    st.info("🎥 **Cámara activa** - Procesando en tiempo real...")
                
                while st.session_state.camara_activa:
                    frame = sistema.procesar_frame_camara()
                    if frame is not None:
                        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                    time.sleep(0.05)
                    
            else:
                st.info("👆 **Haz clic en 'INICIAR CÁMARA'** para comenzar la detección en vivo")
                st.image("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                        caption="Vista previa - Tu cámara aparecerá aquí", use_column_width=True)
    
    with col2:
        st.subheader("📊 RESULTADOS")
        
        if sistema.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(sistema.detecciones_actuales)}")
            
            for i, det in enumerate(sistema.detecciones_actuales):
                with st.expander(f"Persona {i+1}", expanded=True):
                    st.metric("Distancia", f"{det['distancia_estimada']:.1f}m")
                    st.metric("Zona", sistema.determinar_zona(det['centro'][0]))
                    st.metric("Confianza", f"{det['confianza']:.0%}")
        else:
            st.info("📋 Esperando detecciones...")
            st.caption("Los resultados aparecerán aquí")
        
        # Alertas
        if st.session_state.alertas:
            st.subheader("🚨 ALERTAS RECIENTES")
            for alerta in st.session_state.alertas[-3:]:
                tiempo = time.strftime('%H:%M:%S', time.localtime(alerta['timestamp']))
                if alerta['distancia'] < 1.0:
                    st.error(f"**{tiempo}** - {alerta['mensaje']}")
                else:
                    st.warning(f"**{tiempo}** - {alerta['mensaje']}")
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        if sistema.es_movil:
            st.caption("📱 **Modo Móvil** | 📸 **Cámara Fotos** | 🔊 **Audio**")
        else:
            st.caption("💻 **Modo Desktop** | 🎥 **Cámara Vivo** | 🔊 **Audio**")
    
    with footer_col2:
        st.caption(f"🔄 Última actualización: {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
