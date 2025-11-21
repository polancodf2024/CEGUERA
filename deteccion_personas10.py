# deteccion_personas_streamlit_cloud.py
import cv2
import numpy as np
import time
import pygame
import os
import streamlit as st
from PIL import Image
import tempfile
import threading

class SistemaDeteccionPersonasStreamlit:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.frame_actual = None
        self.ultima_alerta = 0
        self.cooldown = 2.0
        self.ejecutando = False
        self.cap = None
        self.video_file = None
        
        # Detectar entorno
        self.modo_cloud = self.detectar_entorno_cloud()
        self.es_dispositivo_movil = self.detectar_dispositivo_movil()
        
        self.audio_disponible = False
        self.sonidos = {}
        
        # Solo inicializar audio si NO estamos en Cloud y NO es móvil
        if not self.modo_cloud and not self.es_dispositivo_movil:
            self.inicializar_audio_mejorado()
        
        print("✅ Sistema Streamlit inicializado correctamente")
        print(f"🔍 Modo Cloud: {self.modo_cloud}")
        print(f"📱 Dispositivo Móvil: {self.es_dispositivo_movil}")
        print(f"🔊 Audio disponible: {self.audio_disponible}")
        
    def detectar_entorno_cloud(self):
        """Detecta si estamos en Streamlit Cloud de manera más precisa"""
        cloud_indicators = [
            'STREAMLIT_SHARING', 'STREAMLIT_SERVER_HEADLESS', 
            'STREAMLIT_SERVER_ADDRESS', 'STREAMLIT_DEPLOYMENT'
        ]
        
        # Verificar variables de entorno
        for var in cloud_indicators:
            if os.getenv(var):
                print(f"🔍 Detectado entorno Cloud: {var}")
                return True
        
        # Verificar si estamos en un entorno sin dispositivos
        try:
            # Intentar acceder a la cámara
            cap_test = cv2.VideoCapture(0)
            if not cap_test.isOpened():
                print("🔍 No se puede acceder a cámara - modo Cloud asumido")
                return True
            cap_test.release()
        except Exception as e:
            print(f"🔍 Error accediendo a cámara - modo Cloud asumido: {e}")
            return True
        
        return False
    
    def detectar_dispositivo_movil(self):
        """Detecta si el usuario accede desde un dispositivo móvil"""
        try:
            # En Streamlit Cloud, asumimos que podría ser móvil
            # En la práctica, necesitaríamos el user agent real
            return self.modo_cloud
        except:
            return False
        
    def inicializar_deteccion_avanzada(self):
        """Inicializa múltiples métodos de detección"""
        print("🔍 Inicializando detectores...")
        self.detectores = []
        
        # 1. Haar Cascade para cuerpo completo
        try:
            haar_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(haar_path):
                detector_haar = cv2.CascadeClassifier(haar_path)
                if not detector_haar.empty():
                    self.detectores.append(('haar_fullbody', detector_haar))
                    print("✅ Haar Fullbody cargado")
            else:
                print("⚠️  Haar Fullbody no encontrado")
        except Exception as e:
            print(f"⚠️  Error Haar Fullbody: {e}")
        
        # 2. Haar Cascade para cuerpo superior
        try:
            haar_upper_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            if os.path.exists(haar_upper_path):
                detector_upper = cv2.CascadeClassifier(haar_upper_path)
                if not detector_upper.empty():
                    self.detectores.append(('haar_upperbody', detector_upper))
                    print("✅ Haar Upperbody cargado")
            else:
                print("⚠️  Haar Upperbody no encontrado")
        except Exception as e:
            print(f"⚠️  Error Haar Upperbody: {e}")
            
        # 3. Detección por movimiento (siempre disponible)
        self.frame_anterior = None
        self.detectores.append(('movimiento', None))
        print("✅ Detección por movimiento disponible")
        
        if not self.detectores:
            print("❌ No hay detectores disponibles")
        else:
            print(f"✅ Total detectores: {len(self.detectores)}")
            
    def inicializar_audio_mejorado(self):
        """Configura audio mejorado - SOLO en escritorio local"""
        if self.modo_cloud or self.es_dispositivo_movil:
            print("🔇 Audio desactivado para Cloud/Móvil")
            return
            
        print("🔊 Inicializando audio mejorado...")
        
        try:
            if os.name == 'posix':
                os.environ['SDL_AUDIODRIVER'] = 'pulse'
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
            
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            if pygame.mixer.get_init():
                self.audio_disponible = True
                print(f"✅ Audio inicializado")
                self.crear_sonidos_mejorados()
            else:
                print("❌ No se pudo inicializar el mixer de pygame")
                
        except Exception as e:
            print(f"❌ Error inicializando audio: {e}")
            
    def crear_sonidos_mejorados(self):
        """Crea sonidos más robustos y audibles"""
        if not self.audio_disponible:
            return
            
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
                wave3 = 0.2 * np.sin(2 * np.pi * (freq1 * 2) * t)
                wave = wave1 + wave2 + wave3
                
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
                print(f"✅ Sonido para {zona} creado")
                
        except Exception as e:
            print(f"❌ Error creando sonidos: {e}")
    
    def reproducir_sonido(self, zona):
        """Reproduce sonido en un hilo separado"""
        if not self.audio_disponible or zona not in self.sonidos:
            return
            
        def play_sound():
            try:
                pygame.mixer.stop()
                self.sonidos[zona].play()
                pygame.time.wait(int(800))
            except Exception as e:
                print(f"❌ Error reproduciendo sonido: {e}")
        
        sound_thread = threading.Thread(target=play_sound)
        sound_thread.daemon = True
        sound_thread.start()
    
    def probar_audio(self):
        """Función para probar todos los sonidos"""
        if not self.audio_disponible:
            return "🔇 Audio no disponible"
            
        try:
            for zona in ['izquierda', 'frente', 'derecha', 'alerta']:
                if zona in self.sonidos:
                    self.reproducir_sonido(zona)
                    time.sleep(0.9)
            return "✅ Sonidos probados correctamente"
        except Exception as e:
            return f"❌ Error probando audio: {e}"
    
    def configurar_parametros(self):
        """Configura parámetros del sistema"""
        self.umbral_confianza = 0.3
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        self.calibracion_distancia = {
            'ancho_referencia_cerca': 300,
            'ancho_referencia_lejos': 80, 
            'distancia_referencia_cerca': 0.5,
            'distancia_referencia_lejos': 2.0,
            'factor_ajuste_camara': 0.3,
            'ancho_base_cerca': 300,
            'ancho_base_lejos': 80
        }
        
        self.zonas = {
            'izquierda': (0, self.ancho_pantalla * 0.4),
            'frente': (self.ancho_pantalla * 0.4, self.ancho_pantalla * 0.6),
            'derecha': (self.ancho_pantalla * 0.6, self.ancho_pantalla)
        }

    # ===== FUNCIONES DE DETECCIÓN =====
    
    def detectar_con_haar(self, frame, detector, nombre):
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
                    'distancia_estimada': self.estimar_distancia_corregida(w, h),
                    'detector': nombre
                })
                
            return resultados
        except Exception as e:
            print(f"❌ Error en detectar_con_haar: {e}")
            return []
    
    def detectar_por_movimiento(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.frame_anterior is None:
                self.frame_anterior = gray
                return []
            
            frame_diff = cv2.absdiff(self.frame_anterior, gray)
            thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=3)
            thresh = cv2.erode(thresh, None, iterations=1)
            
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            resultados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(contorno)
                    
                    relacion = h / w if w > 0 else 0
                    if 1.2 < relacion < 4.0:
                        centro_x = x + w//2
                        
                        resultados.append({
                            'bbox': (x, y, x + w, y + h),
                            'centro': (centro_x, y + h//2),
                            'confianza': min(0.6, area / 3000),
                            'distancia_estimada': self.estimar_distancia_corregida(w, h),
                            'detector': 'movimiento'
                        })
            
            self.frame_anterior = gray
            return resultados
        except Exception as e:
            print(f"❌ Error en detectar_por_movimiento: {e}")
            return []
    
    def estimar_distancia_corregida(self, ancho_persona, alto_persona):
        try:
            cal = self.calibracion_distancia
            ancho_efectivo = ancho_persona
            
            if ancho_efectivo <= 0:
                return 5.0
            
            if ancho_efectivo >= cal['ancho_referencia_cerca']:
                proporcion = cal['ancho_referencia_cerca'] / ancho_efectivo
                distancia = cal['distancia_referencia_cerca'] * proporcion
            elif ancho_efectivo <= cal['ancho_referencia_lejos']:
                proporcion = cal['ancho_referencia_lejos'] / ancho_efectivo
                distancia = cal['distancia_referencia_lejos'] * proporcion
            else:
                rango_pixeles = cal['ancho_referencia_cerca'] - cal['ancho_referencia_lejos']
                rango_distancias = cal['distancia_referencia_lejos'] - cal['distancia_referencia_cerca']
                pixeles_desde_lejos = ancho_efectivo - cal['ancho_referencia_lejos']
                proporcion = pixeles_desde_lejos / rango_pixeles
                distancia = cal['distancia_referencia_cerca'] + (rango_distancias * (1 - proporcion))
            
            distancia_corregida = distancia * cal['factor_ajuste_camara']
            return max(0.1, min(10.0, distancia_corregida))
        except Exception as e:
            print(f"❌ Error estimando distancia: {e}")
            return 2.0
    
    def calibrar_con_distancia_real(self, distancia_real_cm):
        if not self.detecciones_actuales:
            return False
            
        try:
            persona = min(self.detecciones_actuales, key=lambda x: x['distancia_estimada'])
            ancho_pixels = persona['bbox'][2] - persona['bbox'][0]
            distancia_real_m = distancia_real_cm / 100.0
            
            if distancia_real_m <= 1.0:
                self.calibracion_distancia['ancho_referencia_cerca'] = ancho_pixels
                self.calibracion_distancia['distancia_referencia_cerca'] = distancia_real_m
                return f"✅ Punto cercano: {ancho_pixels}px = {distancia_real_cm}cm"
            else:
                self.calibracion_distancia['ancho_referencia_lejos'] = ancho_pixels
                self.calibracion_distancia['distancia_referencia_lejos'] = distancia_real_m
                return f"✅ Punto lejano: {ancho_pixels}px = {distancia_real_cm}cm"
        except Exception as e:
            return f"❌ Error en calibración: {e}"
    
    def auto_calibrar_con_factor(self):
        if not self.detecciones_actuales:
            return "❌ No hay detecciones para calibrar"
            
        try:
            persona = min(self.detecciones_actuales, key=lambda x: x['distancia_estimada'])
            distancia_reportada = persona['distancia_estimada']
            factor_sugerido = 0.5 / distancia_reportada
            factor_actual = self.calibracion_distancia['factor_ajuste_camara']
            nuevo_factor = (factor_actual + factor_sugerido) / 2
            self.calibracion_distancia['factor_ajuste_camara'] = max(0.1, min(2.0, nuevo_factor))
            return f"🔧 Auto-calibrado: factor {nuevo_factor:.3f}"
        except Exception as e:
            return f"❌ Error en auto-calibración: {e}"

    def reset_calibracion(self):
        try:
            self.calibracion_distancia['ancho_referencia_cerca'] = self.calibracion_distancia['ancho_base_cerca']
            self.calibracion_distancia['ancho_referencia_lejos'] = self.calibracion_distancia['ancho_base_lejos']
            self.calibracion_distancia['factor_ajuste_camara'] = 0.3
            return "✅ Calibración reseteada"
        except Exception as e:
            return f"❌ Error reseteando calibración: {e}"
    
    def detectar_personas(self, frame):
        todas_detecciones = []
        
        for nombre, detector in self.detectores:
            try:
                if nombre == 'movimiento':
                    detecciones = self.detectar_por_movimiento(frame)
                else:
                    detecciones = self.detectar_con_haar(frame, detector, nombre)
                todas_detecciones.extend(detecciones)
            except Exception as e:
                print(f"⚠️  Error en detector {nombre}: {e}")
        
        return self.filtrar_duplicados(todas_detecciones)
    
    def filtrar_duplicados(self, detecciones, umbral_solapamiento=0.5):
        if not detecciones:
            return []
            
        try:
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
        except Exception as e:
            print(f"❌ Error filtrando duplicados: {e}")
            return detecciones
    
    def determinar_zona(self, centro_x):
        for zona, (inicio, fin) in self.zonas.items():
            if inicio <= centro_x <= fin:
                return zona
        return 'frente'
    
    def generar_alerta(self, detecciones):
        tiempo_actual = time.time()
        if tiempo_actual - self.ultima_alerta < self.cooldown or not detecciones:
            return
        
        try:
            persona_cercana = min(detecciones, key=lambda x: x['distancia_estimada'])
            distancia = persona_cercana['distancia_estimada']
            zona = self.determinar_zona(persona_cercana['centro'][0])
            
            if distancia < 2.0:
                mensaje = ""
                if self.audio_disponible:
                    try:
                        if distancia < 0.6:
                            for _ in range(2):
                                self.reproducir_sonido('alerta')
                                time.sleep(0.2)
                            mensaje = f"🚨 MUY CERCA! {distancia:.1f}m - {zona.upper()}"
                        else:
                            self.reproducir_sonido(zona)
                            mensaje = f"🔊 Persona: {distancia:.1f}m - {zona}"
                    except Exception as e:
                        mensaje = f"🔇 [AUDIO ERROR] Persona: {distancia:.1f}m - {zona}"
                else:
                    if distancia < 0.6:
                        mensaje = f"🚨 ALERTA: Persona MUY CERCA a {distancia:.1f}m - {zona.upper()}"
                    else:
                        mensaje = f"🔊 Persona detectada: {distancia:.1f}m - {zona}"
                
                if 'alertas' in st.session_state:
                    if len(st.session_state.alertas) >= 10:
                        st.session_state.alertas.pop(0)
                    st.session_state.alertas.append({
                        'timestamp': time.time(),
                        'mensaje': mensaje,
                        'distancia': distancia,
                        'zona': zona
                    })
                
                print(f"👤 {mensaje}")
                self.ultima_alerta = tiempo_actual
        except Exception as e:
            print(f"❌ Error generando alerta: {e}")
    
    def dibujar_interfaz(self, frame, detecciones):
        try:
            # Redimensionar frame si es necesario
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
            status_text = f"Personas: {len(detecciones)} | Audio: {'✅' if self.audio_disponible else '❌'}"
            cv2.putText(frame, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            for i, det in enumerate(detecciones):
                x1, y1, x2, y2 = det['bbox']
                distancia = det['distancia_estimada']
                zona = self.determinar_zona(det['centro'][0])
                
                if distancia < 0.6:
                    color = (0, 0, 255)
                elif distancia < 1.2:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{i+1}: {distancia:.1f}m ({zona})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, det['centro'], 5, color, -1)
            
            return frame
        except Exception as e:
            print(f"❌ Error dibujando interfaz: {e}")
            return frame

    # ===== FUNCIONALIDAD DE CÁMARA =====
    def iniciar_camara(self):
        """Inicia la cámara - solo en escritorio local"""
        if self.modo_cloud or self.es_dispositivo_movil:
            return False, "❌ La cámara no está disponible en Cloud/Móvil"
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                return False, "❌ No se pudo acceder a la cámara"
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.ejecutando = True
            return True, "✅ Cámara iniciada correctamente"
        except Exception as e:
            return False, f"❌ Error iniciando cámara: {e}"

    def detener_camara(self):
        self.ejecutando = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def procesar_frame_camara(self):
        if not self.cap or not self.ejecutando:
            return None
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_flipped = cv2.flip(frame_rgb, 1)
            detecciones = self.detectar_personas(frame_flipped)
            self.generar_alerta(detecciones)
            frame_procesado = self.dibujar_interfaz(frame_flipped, detecciones)
            self.detecciones_actuales = detecciones
            return frame_procesado
        except Exception as e:
            print(f"❌ Error procesando frame cámara: {e}")
            return None

    # ===== FUNCIONALIDAD DE ARCHIVOS =====
    def procesar_video(self, video_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
            
            self.cap = cv2.VideoCapture(video_path)
            self.ejecutando = True
            frames_procesados = []
            
            while self.ejecutando:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_procesado = self.procesar_frame_completo(frame_rgb)
                frames_procesados.append(frame_procesado)
                
            self.cap.release()
            try:
                os.unlink(video_path)
            except:
                pass
            return frames_procesados
            
        except Exception as e:
            st.error(f"❌ Error procesando video: {e}")
            return []

    def procesar_imagen(self, image_file):
        try:
            image = Image.open(image_file)
            frame = np.array(image)
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_procesado = self.procesar_frame_completo(frame_rgb)
            return frame_procesado
        except Exception as e:
            st.error(f"❌ Error procesando imagen: {e}")
            return None

    def procesar_frame_completo(self, frame):
        try:
            detecciones = self.detectar_personas(frame)
            self.generar_alerta(detecciones)
            frame_procesado = self.dibujar_interfaz(frame, detecciones)
            self.detecciones_actuales = detecciones
            return frame_procesado
        except Exception as e:
            print(f"❌ Error procesando frame completo: {e}")
            return frame

# Configuración de Streamlit
def main():
    st.set_page_config(
        page_title="Sistema de Detección de Personas",
        page_icon="👤",
        layout="wide"
    )
    
    # Inicializar sesión
    if 'sistema' not in st.session_state:
        st.session_state.sistema = SistemaDeteccionPersonasStreamlit()
        st.session_state.alertas = []
        st.session_state.camara_activa = False
    
    sistema = st.session_state.sistema
    
    # Header mejorado con información clara
    st.title("👤 Sistema de Detección de Personas")
    
    # Banner informativo según el entorno
    if sistema.modo_cloud:
        st.warning("""
        ☁️ **MODO STREAMLIT CLOUD** 
        - 📁 **Sube imágenes o videos** para procesar
        - 🔇 **Audio no disponible** en este entorno  
        - ❌ **Cámara en vivo no disponible**
        - ✅ **Detección funciona** con archivos subidos
        """)
    elif sistema.es_dispositivo_movil:
        st.info("""
        📱 **MODO DISPOSITIVO MÓVIL**
        - 📁 **Sube archivos** desde tu galería
        - 📸 **Toma fotos/videos** y súbelos
        - ✅ **Detección funciona** perfectamente
        """)
    else:
        st.success("""
        💻 **MODO ESCRITORIO LOCAL**
        - 🎥 **Cámara en vivo** disponible
        - 🔊 **Audio espacial** activado
        - 📁 **Subir archivos** también disponible
        """)
    
    st.markdown("---")
    
    # Sidebar mejorado
    with st.sidebar:
        st.header("⚙️ CONFIGURACIÓN")
        
        # Estado del sistema
        st.subheader("📊 ESTADO DEL SISTEMA")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entorno", "☁️ Cloud" if sistema.modo_cloud else "📱 Móvil" if sistema.es_dispositivo_movil else "💻 Escritorio")
        with col2:
            st.metric("Audio", "❌" if sistema.modo_cloud or sistema.es_dispositivo_movil else "✅")
        
        # Solo mostrar opción de cámara si NO estamos en Cloud
        if not sistema.modo_cloud:
            st.subheader("🎥 CÁMARA EN VIVO")
            if not st.session_state.camara_activa:
                if st.button("🎥 Iniciar Cámara", type="primary", use_container_width=True):
                    exito, mensaje = sistema.iniciar_camara()
                    if exito:
                        st.session_state.camara_activa = True
                        st.success(mensaje)
                        st.rerun()
                    else:
                        st.error(mensaje)
            else:
                if st.button("⏹️ Detener Cámara", use_container_width=True):
                    sistema.detener_camara()
                    st.session_state.camara_activa = False
                    st.rerun()
        
        # SUBIR ARCHIVOS (siempre disponible)
        st.subheader("📁 SUBIR ARCHIVOS")
        tipo_archivo = st.radio(
            "Tipo de archivo:",
            ["Imagen", "Video"],
            horizontal=True
        )
        
        if tipo_archivo == "Imagen":
            archivo_subido = st.file_uploader(
                "Sube una imagen", 
                type=['jpg', 'jpeg', 'png'],
                help="Sube una imagen para detectar personas"
            )
        else:
            archivo_subido = st.file_uploader(
                "Sube un video", 
                type=['mp4', 'avi', 'mov'],
                help="Sube un video para detectar personas"
            )
        
        # Controles de audio (solo escritorio local)
        if not sistema.modo_cloud and not sistema.es_dispositivo_movil:
            st.markdown("---")
            st.subheader("🔊 AUDIO")
            if st.button("🎵 Probar Sonidos", type="secondary", use_container_width=True):
                resultado = sistema.probar_audio()
                st.info(resultado)
        
        # Controles de calibración
        st.markdown("---")
        st.subheader("🎯 CALIBRACIÓN")
        
        if st.button("🔄 Auto-calibrar", use_container_width=True):
            if sistema.detecciones_actuales:
                resultado = sistema.auto_calibrar_con_factor()
                st.success(resultado)
            else:
                st.warning("Toma una foto con personas para calibrar")
        
        # Factor de calibración
        factor_actual = sistema.calibracion_distancia['factor_ajuste_camara']
        nuevo_factor = st.slider(
            "Factor de distancia:", 
            0.1, 2.0, float(factor_actual), 0.1,
            help="Ajusta si las distancias no son precisas"
        )
        if nuevo_factor != factor_actual:
            sistema.calibracion_distancia['factor_ajuste_camara'] = nuevo_factor
            st.rerun()
        
        # Calibración manual
        st.caption("Calibración Manual")
        distancia_calibracion = st.number_input("Distancia real (cm):", min_value=10, max_value=300, value=50, step=10)
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            if st.button("📏 Calibrar", use_container_width=True):
                if sistema.detecciones_actuales:
                    resultado = sistema.calibrar_con_distancia_real(distancia_calibracion)
                    st.success(resultado)
                else:
                    st.warning("Primero procesa una imagen con personas")
        with col_cal2:
            if st.button("🔄 Reset", use_container_width=True):
                resultado = sistema.reset_calibracion()
                st.info(resultado)
    
    # ÁREA PRINCIPAL MEJORADA
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎬 VISTA PREVIA")
        
        # MODO CÁMARA (solo local)
        if not sistema.modo_cloud and st.session_state.camara_activa:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with status_placeholder:
                st.info("🎥 **Cámara activa** - Procesando en tiempo real...")
            
            while st.session_state.camara_activa and not sistema.modo_cloud:
                frame_procesado = sistema.procesar_frame_camara()
                if frame_procesado is not None:
                    frame_placeholder.image(frame_procesado, channels="RGB", use_column_width=True)
                time.sleep(0.03)
        
        # MODO ARCHIVOS (para Cloud y móvil)
        else:
            # Usar archivo subido del sidebar
            archivo_subido = None
            if 'tipo_archivo' in locals() and tipo_archivo == "Imagen":
                archivo_subido = archivo_subido
            elif 'tipo_archivo' in locals() and tipo_archivo == "Video":
                archivo_subido = archivo_subido
            
            if archivo_subido is not None:
                st.success(f"📁 **Archivo cargado:** {archivo_subido.name}")
                
                if tipo_archivo == "Imagen":
                    with st.spinner("🔍 Procesando imagen..."):
                        frame_procesado = sistema.procesar_imagen(archivo_subido)
                    
                    if frame_procesado is not None:
                        st.image(frame_procesado, channels="RGB", use_column_width=True)
                        
                        # Mostrar resultados
                        if sistema.detecciones_actuales:
                            st.success(f"✅ **{len(sistema.detecciones_actuales)} persona(s) detectada(s)**")
                        else:
                            st.info("❌ No se detectaron personas")
                
                else:  # Video
                    with st.spinner("🎬 Procesando video..."):
                        frames = sistema.procesar_video(archivo_subido)
                    
                    if frames:
                        st.success(f"✅ **Video procesado:** {len(frames)} frames analizados")
                        st.image(frames[-1], channels="RGB", use_column_width=True, caption="Último frame procesado")
                        
                        # Mostrar estadísticas del video
                        if sistema.detecciones_actuales:
                            st.info(f"📊 **En el último frame:** {len(sistema.detecciones_actuales)} persona(s)")
                    else:
                        st.error("❌ Error al procesar el video")
            
            else:
                # Pantalla de bienvenida según el entorno
                if sistema.modo_cloud:
                    st.info("""
                    **👆 PARA COMENZAR:**
                    
                    1. **Selecciona** Imagen o Video en el panel izquierdo
                    2. **Sube** un archivo desde tu computadora
                    3. **Espera** a que se procese
                    4. **Ve** los resultados y alertas
                    
                    💡 **Consejo:** Usa videos cortos (menos de 10MB) para mejor rendimiento
                    """)
                    
                    # Ejemplo de imagen para probar
                    st.markdown("---")
                    st.subheader("🖼️ Ejemplo de Imagen para Probar")
                    st.image("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", 
                            caption="Imagen de ejemplo (Lena) - Sube una imagen similar con personas",
                            use_column_width=True)
                            
                else:
                    st.info("""
                    **👆 SELECCIONA UN MODO:**
                    
                    - **📁 Subir Archivo**: Imágenes o videos
                    - **🎥 Cámara en Vivo**: Si estás en escritorio local
                    
                    💡 **Consejo:** En Cloud, usa la opción de subir archivos
                    """)
    
    with col2:
        st.subheader("📊 RESULTADOS")
        
        if sistema.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(sistema.detecciones_actuales)}")
            
            for i, det in enumerate(sistema.detecciones_actuales):
                with st.expander(f"Persona {i+1}", expanded=True):
                    st.metric("Distancia", f"{det['distancia_estimada']:.2f}m")
                    st.metric("Zona", det.get('zona', sistema.determinar_zona(det['centro'][0])))
                    st.metric("Confianza", f"{det['confianza']:.1%}")
                    st.caption(f"Detector: {det['detector']}")
                    
                    # Información adicional
                    ancho = det['bbox'][2] - det['bbox'][0]
                    alto = det['bbox'][3] - det['bbox'][1]
                    st.caption(f"Tamaño: {ancho}×{alto}px")
        else:
            st.info("📋 **Esperando datos...**")
            st.caption("Los resultados aparecerán aquí después del procesamiento")
        
        # Alertas recientes
        if st.session_state.alertas:
            st.subheader("🚨 ALERTAS")
            for alerta in st.session_state.alertas[-3:]:
                tiempo = time.strftime('%H:%M:%S', time.localtime(alerta['timestamp']))
                if alerta['distancia'] < 0.6:
                    st.error(f"**{tiempo}** - {alerta['mensaje']}")
                else:
                    st.warning(f"**{tiempo}** - {alerta['mensaje']}")
        
        # Estadísticas del sistema
        st.markdown("---")
        st.subheader("📈 ESTADÍSTICAS")
        if st.session_state.alertas:
            st.metric("Alertas Totales", len(st.session_state.alertas))
            if st.session_state.alertas:
                ultima = st.session_state.alertas[-1]
                st.metric("Última Distancia", f"{ultima['distancia']:.2f}m")
                st.metric("Zona", ultima['zona'])
        else:
            st.info("No hay estadísticas aún")
    
    # Footer informativo
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        if sistema.modo_cloud:
            st.caption("**Sistema de Detección** | ☁️ **Streamlit Cloud**")
        elif sistema.es_dispositivo_movil:
            st.caption("**Sistema de Detección** | 📱 **Dispositivo Móvil**")
        else:
            st.caption("**Sistema de Detección** | 💻 **Escritorio Local**")
    
    with footer_col2:
        if sistema.modo_cloud or sistema.es_dispositivo_movil:
            st.caption("📁 **Modo Archivos** | 🔇 **Sin Audio**")
        else:
            st.caption("🎥 **Cámara + 🔊 Audio** | 📁 **Archivos**")
    
    with footer_col3:
        st.caption(f"Detectores: {len(sistema.detectores)} | v2.0")

if __name__ == "__main__":
    main()
