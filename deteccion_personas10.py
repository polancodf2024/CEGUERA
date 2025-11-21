# deteccion_personas7.py
import cv2
import numpy as np
import time
import pygame
import os
import streamlit as st
import threading
from queue import Queue
import tempfile

class SistemaDeteccionPersonasStreamlit:
    def __init__(self):
        self.inicializar_deteccion_avanzada()
        self.inicializar_audio_mejorado()
        self.configurar_parametros()
        self.detecciones_actuales = []
        self.frame_actual = None
        self.ultima_alerta = 0
        self.cooldown = 2.0
        self.ejecutando = False
        self.cap = None
        print("✅ Sistema Streamlit inicializado correctamente")
        
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
        except Exception as e:
            print(f"⚠️  Error Haar Upperbody: {e}")
            
        # 3. Detección por movimiento (siempre disponible)
        self.frame_anterior = None
        self.detectores.append(('movimiento', None))
        print("✅ Detección por movimiento disponible")
        
        if not self.detectores:
            print("❌ No hay detectores disponibles")
            
    def inicializar_audio_mejorado(self):
        """Configura audio mejorado y más confiable"""
        print("🔊 Inicializando audio mejorado...")
        self.audio_disponible = False
        self.sonidos = {}
        
        try:
            # Inicializar pygame solo si no está inicializado
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=22050,
                    size=-16,
                    channels=2,
                    buffer=512
                )
                
            if pygame.mixer.get_init():
                self.audio_disponible = True
                print(f"✅ Audio inicializado: {pygame.mixer.get_init()}")
                self.crear_sonidos_mejorados()
            else:
                print("❌ No se pudo inicializar el mixer de pygame")
                
        except Exception as e:
            print(f"❌ Error crítico inicializando audio: {e}")
            self.audio_disponible = False
            
    def crear_sonidos_mejorados(self):
        """Crea sonidos más robustos y audibles"""
        try:
            sample_rate = 22050
            duration = 0.5
            
            tonos = {
                'izquierda': (440, 330),
                'derecha': (523, 392),
                'frente': (659, 494)
            }
            
            for zona, (freq1, freq2) in tonos.items():
                samples = int(sample_rate * duration)
                t = np.linspace(0, duration, samples, False)
                
                wave1 = 0.5 * np.sin(2 * np.pi * freq1 * t)
                wave2 = 0.3 * np.sin(2 * np.pi * freq2 * t)
                wave = wave1 + wave2
                
                envelope = np.ones_like(t)
                attack = int(0.1 * sample_rate)
                release = int(0.2 * sample_rate)
                
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                
                wave *= envelope
                
                wave_int = (wave * 32767).astype(np.int16)
                wave_stereo = np.column_stack((wave_int, wave_int))
                
                sound = pygame.sndarray.make_sound(wave_stereo)
                self.sonidos[zona] = sound
                print(f"✅ Sonido para {zona} creado: {freq1}Hz + {freq2}Hz")
                
        except Exception as e:
            print(f"❌ Error creando sonidos: {e}")
            self.audio_disponible = False
    
    def configurar_parametros(self):
        """Configura parámetros del sistema con mejor calibración"""
        self.umbral_confianza = 0.3
        self.ancho_pantalla = 640
        self.alto_pantalla = 480
        
        # 🔥 CALIBRACIÓN INICIAL MÁS PRECISA
        # Basado en tu caso: 50cm reales = 1.60m reportados (factor ~0.31)
        self.calibracion_distancia = {
            'ancho_referencia_cerca': 300,    # Valor inicial
            'ancho_referencia_lejos': 80,     # Valor inicial  
            'distancia_referencia_cerca': 0.5, # 50cm
            'distancia_referencia_lejos': 2.0, # 2m
            'factor_ajuste_camara': 0.3,      # 🔥 FACTOR REDUCIDO (antes 1.2)
            'ancho_base_cerca': 300,          # Para reset
            'ancho_base_lejos': 80            # Para reset
        }
        
        # Zonas de la pantalla
        self.zonas = {
            'izquierda': (0, self.ancho_pantalla * 0.4),
            'frente': (self.ancho_pantalla * 0.4, self.ancho_pantalla * 0.6),
            'derecha': (self.ancho_pantalla * 0.6, self.ancho_pantalla)
        }
        
    def detectar_con_haar(self, frame, detector, nombre):
        """Detección usando Haar Cascade específico"""
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
    
    def detectar_por_movimiento(self, frame):
        """Detección mejorada por movimiento"""
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
    
    def estimar_distancia_corregida(self, ancho_persona, alto_persona):
        """🔥 DISTANCIA CORREGIDA - Fórmula simplificada y precisa"""
        cal = self.calibracion_distancia
        
        # Usar principalmente el ancho (más estable que alto)
        ancho_efectivo = ancho_persona
        
        if ancho_efectivo <= 0:
            return 5.0
        
        # 🔥 FÓRMULA SIMPLIFICADA Y MÁS PRECISA
        # Basada en relación inversa simple: más ancho = más cerca
        if ancho_efectivo >= cal['ancho_referencia_cerca']:
            # Muy cerca - extrapolación
            proporcion = cal['ancho_referencia_cerca'] / ancho_efectivo
            distancia = cal['distancia_referencia_cerca'] * proporcion
        elif ancho_efectivo <= cal['ancho_referencia_lejos']:
            # Muy lejos - extrapolación
            proporcion = cal['ancho_referencia_lejos'] / ancho_efectivo
            distancia = cal['distancia_referencia_lejos'] * proporcion
        else:
            # Interpolación lineal
            rango_pixeles = cal['ancho_referencia_cerca'] - cal['ancho_referencia_lejos']
            rango_distancias = cal['distancia_referencia_lejos'] - cal['distancia_referencia_cerca']
            
            pixeles_desde_lejos = ancho_efectivo - cal['ancho_referencia_lejos']
            proporcion = pixeles_desde_lejos / rango_pixeles
            
            distancia = cal['distancia_referencia_cerca'] + (rango_distancias * (1 - proporcion))
        
        # 🔥 APLICAR FACTOR DE CORRECCIÓN
        distancia_corregida = distancia * cal['factor_ajuste_camara']
        
        return max(0.1, min(10.0, distancia_corregida))
    
    def calibrar_con_distancia_real(self, distancia_real_cm):
        """🔥 CALIBRACIÓN PRECISA - Usando distancia real en cm"""
        if not self.detecciones_actuales:
            return False
            
        persona = min(self.detecciones_actuales, key=lambda x: x['distancia_estimada'])
        ancho_pixels = persona['bbox'][2] - persona['bbox'][0]
        distancia_real_m = distancia_real_cm / 100.0
        
        print(f"🔧 Calibración: {distancia_real_cm}cm -> {ancho_pixels}px")
        
        if distancia_real_m <= 1.0:  # Menos de 1 metro = punto cercano
            self.calibracion_distancia['ancho_referencia_cerca'] = ancho_pixels
            self.calibracion_distancia['distancia_referencia_cerca'] = distancia_real_m
            return f"✅ Punto cercano: {ancho_pixels}px = {distancia_real_cm}cm"
        else:  # Más de 1 metro = punto lejano
            self.calibracion_distancia['ancho_referencia_lejos'] = ancho_pixels
            self.calibracion_distancia['distancia_referencia_lejos'] = distancia_real_m
            return f"✅ Punto lejano: {ancho_pixels}px = {distancia_real_cm}cm"
    
    def auto_calibrar_con_factor(self):
        """🔥 CALIBRACIÓN AUTOMÁTICA basada en la discrepancia actual"""
        if not self.detecciones_actuales:
            return "❌ No hay detecciones para calibrar"
            
        persona = min(self.detecciones_actuales, key=lambda x: x['distancia_estimada'])
        distancia_reportada = persona['distancia_estimada']
        ancho_actual = persona['bbox'][2] - persona['bbox'][0]
        
        # Calcular nuevo factor basado en la discrepancia
        # Si reporta 1.60m pero estás a 0.50m, necesitamos factor ~0.31
        factor_sugerido = 0.5 / distancia_reportada  # 50cm / 160cm
        
        # Suavizar el cambio
        factor_actual = self.calibracion_distancia['factor_ajuste_camara']
        nuevo_factor = (factor_actual + factor_sugerido) / 2
        
        self.calibracion_distancia['factor_ajuste_camara'] = max(0.1, min(2.0, nuevo_factor))
        
        return f"🔧 Auto-calibrado: factor {nuevo_factor:.3f} (de {distancia_reportada:.1f}m a ~0.5m)"

    def reset_calibracion(self):
        """Resetear calibración a valores por defecto"""
        self.calibracion_distancia['ancho_referencia_cerca'] = self.calibracion_distancia['ancho_base_cerca']
        self.calibracion_distancia['ancho_referencia_lejos'] = self.calibracion_distancia['ancho_base_lejos']
        self.calibracion_distancia['factor_ajuste_camara'] = 0.3
        return "✅ Calibración reseteada a valores por defecto"
    
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
                print(f"⚠️  Error en detector {nombre}: {e}")
        
        return self.filtrar_duplicados(todas_detecciones)
    
    def filtrar_duplicados(self, detecciones, umbral_solapamiento=0.5):
        """Elimina detecciones duplicadas"""
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
    
    def generar_alerta(self, detecciones):
        """Genera alertas según las detecciones"""
        tiempo_actual = time.time()
        
        if tiempo_actual - self.ultima_alerta < self.cooldown:
            return
        
        if not detecciones:
            return
        
        persona_cercana = min(detecciones, key=lambda x: x['distancia_estimada'])
        distancia = persona_cercana['distancia_estimada']
        zona = self.determinar_zona(persona_cercana['centro'][0])
        
        if distancia < 2.0:  # 🔥 Umbral reducido
            mensaje = ""
            if self.audio_disponible and self.sonidos:
                try:
                    if distancia < 0.6:  # 🔥 Umbral de "muy cerca" reducido
                        for _ in range(2):
                            if zona in self.sonidos:
                                self.sonidos[zona].play()
                                time.sleep(0.2)
                        mensaje = f"🚨 MUY CERCA! {distancia:.1f}m - {zona.upper()}"
                    else:
                        if zona in self.sonidos:
                            self.sonidos[zona].play()
                        mensaje = f"🔊 Persona: {distancia:.1f}m - {zona}"
                except Exception as e:
                    print(f"⚠️  Error reproduciendo audio: {e}")
                    mensaje = f"🔇 [AUDIO ERROR] Persona: {distancia:.1f}m - {zona}"
            else:
                if distancia < 0.6:
                    mensaje = f"🚨 ALERTA: Persona MUY CERCA a {distancia:.1f}m - {zona.upper()}"
                else:
                    mensaje = f"🔊 Persona detectada: {distancia:.1f}m - {zona}"
            
            # Agregar a alertas de Streamlit
            if 'alertas' in st.session_state:
                if len(st.session_state.alertas) >= 10:
                    st.session_state.alertas.pop(0)
                st.session_state.alertas.append({
                    'timestamp': time.time(),
                    'mensaje': mensaje,
                    'distancia': distancia,
                    'zona': zona
                })
            
            print(f"👤 {mensaje} | Detector: {persona_cercana['detector']}")
            self.ultima_alerta = tiempo_actual
    
    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja interfaz mejorada con información"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.ancho_pantalla, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
        cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if detecciones else (0, 0, 255)
        status_text = f"Personas: {len(detecciones)} | Factor: {self.calibracion_distancia['factor_ajuste_camara']:.3f}"
        cv2.putText(frame, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cal_info = f"Cerca: {self.calibracion_distancia['ancho_referencia_cerca']}px"
        cv2.putText(frame, cal_info, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, det in enumerate(detecciones):
            x1, y1, x2, y2 = det['bbox']
            distancia = det['distancia_estimada']
            zona = self.determinar_zona(det['centro'][0])
            detector = det['detector']
            ancho = x2 - x1
            alto = y2 - y1
            
            # Colores según distancias realistas
            if distancia < 0.6:
                color = (0, 0, 255)  # Rojo - Muy cerca
            elif distancia < 1.2:
                color = (0, 165, 255)  # Naranja - Cerca
            else:
                color = (0, 255, 0)  # Verde - Normal
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{i+1}: {distancia:.1f}m ({zona})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.circle(frame, det['centro'], 5, color, -1)
            
            dim_text = f"{ancho}x{alto}"
            cv2.putText(frame, dim_text, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return frame

    def iniciar_camara(self):
        """Inicia la cámara"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.ejecutando = True
            return True
        except Exception as e:
            print(f"❌ Error iniciando cámara: {e}")
            return False

    def detener_camara(self):
        """Detiene la cámara"""
        self.ejecutando = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def procesar_frame(self):
        """Procesa un frame de la cámara"""
        if not self.cap or not self.ejecutando:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Procesar frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flipped = cv2.flip(frame_rgb, 1)
        
        # Detectar personas
        detecciones = self.detectar_personas(frame_flipped)
        
        # Generar alertas
        self.generar_alerta(detecciones)
        
        # Dibujar interfaz
        frame_procesado = self.dibujar_interfaz(frame_flipped, detecciones)
        
        # Actualizar estado
        self.detecciones_actuales = detecciones
        self.frame_actual = frame_procesado
        
        return frame_procesado

# Configuración de Streamlit
def main():
    st.set_page_config(
        page_title="Sistema de Detección de Personas - CALIBRACIÓN MEJORADA",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🎯 Sistema de Detección - 🔥 CALIBRACIÓN PRECISA")
    st.markdown("---")
    
    # Inicializar sesión
    if 'sistema' not in st.session_state:
        st.session_state.sistema = SistemaDeteccionPersonasStreamlit()
        st.session_state.alertas = []
        st.session_state.camara_activa = False
    
    # Sidebar con controles MEJORADOS
    with st.sidebar:
        st.header("⚙️ CALIBRACIÓN PRECISA")
        
        st.subheader("Control de Cámara")
        if not st.session_state.camara_activa:
            if st.button("🎥 Iniciar Cámara", type="primary"):
                if st.session_state.sistema.iniciar_camara():
                    st.session_state.camara_activa = True
                    st.rerun()
                else:
                    st.error("No se pudo iniciar la cámara")
        else:
            if st.button("⏹️ Detener Cámara"):
                st.session_state.sistema.detener_camara()
                st.session_state.camara_activa = False
                st.rerun()
        
        st.markdown("---")
        st.subheader("🔥 CALIBRACIÓN RÁPIDA")
        
        # Calibración automática
        if st.button("🎯 AUTO-CALIBRAR (50cm)", type="secondary"):
            if st.session_state.sistema.detecciones_actuales:
                resultado = st.session_state.sistema.auto_calibrar_con_factor()
                st.success(resultado)
            else:
                st.error("No hay detecciones para calibrar")
        
        # Calibración manual precisa
        st.subheader("Calibración Manual Precisa")
        distancia_calibracion = st.number_input(
            "Distancia real (cm):",
            min_value=10,
            max_value=300,
            value=50,
            step=10
        )
        
        if st.button(f"📏 Calibrar a {distancia_calibracion}cm"):
            if st.session_state.sistema.detecciones_actuales:
                resultado = st.session_state.sistema.calibrar_con_distancia_real(distancia_calibracion)
                st.success(resultado)
            else:
                st.error("No hay detecciones para calibrar")
        
        st.markdown("---")
        st.subheader("Ajuste Fino del Factor")
        
        # Slider de factor con más precisión
        factor_actual = st.session_state.sistema.calibracion_distancia['factor_ajuste_camara']
        nuevo_factor = st.slider(
            "Factor de Corrección:",
            min_value=0.1,
            max_value=1.0,
            value=float(factor_actual),
            step=0.01,
            format="%.3f",
            key="factor_precision"
        )
        
        if nuevo_factor != factor_actual:
            st.session_state.sistema.calibracion_distancia['factor_ajuste_camara'] = nuevo_factor
            st.info(f"🔧 Factor actualizado: {nuevo_factor:.3f}")
        
        # Botones de ajuste rápido
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➖ Disminuir"):
                nuevo = max(0.1, factor_actual - 0.05)
                st.session_state.sistema.calibracion_distancia['factor_ajuste_camara'] = nuevo
                st.rerun()
        with col2:
            if st.button("➕ Aumentar"):
                nuevo = min(1.0, factor_actual + 0.05)
                st.session_state.sistema.calibracion_distancia['factor_ajuste_camara'] = nuevo
                st.rerun()
        
        # Reset
        if st.button("🔄 Resetear Calibración"):
            resultado = st.session_state.sistema.reset_calibracion()
            st.info(resultado)
            st.rerun()
        
        st.markdown("---")
        st.subheader("Audio")
        if st.button("🔊 Probar Sonidos"):
            try:
                for zona in ['izquierda', 'frente', 'derecha']:
                    if zona in st.session_state.sistema.sonidos:
                        st.session_state.sistema.sonidos[zona].play()
                        time.sleep(0.5)
                st.success("Sonidos probados correctamente")
            except Exception as e:
                st.error(f"Error probando sonidos: {e}")
        
        st.markdown("---")
        st.subheader("Estado del Sistema")
        st.info(f"🔍 Detectores: {len(st.session_state.sistema.detectores)}")
        st.info(f"🔊 Audio: {'✅' if st.session_state.sistema.audio_disponible else '❌'}")
        st.info(f"📷 Cámara: {'✅' if st.session_state.camara_activa else '❌'}")
        st.info(f"🎯 Factor: {st.session_state.sistema.calibracion_distancia['factor_ajuste_camara']:.3f}")
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video en Tiempo Real")
        
        if st.session_state.camara_activa:
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            # Procesar frames en tiempo real
            while st.session_state.camara_activa:
                frame_procesado = st.session_state.sistema.procesar_frame()
                
                if frame_procesado is not None:
                    # Mostrar frame
                    frame_placeholder.image(frame_procesado, channels="RGB", use_column_width=True)
                    
                    # Mostrar alertas recientes
                    if st.session_state.alertas:
                        with alert_placeholder.container():
                            st.subheader("🚨 Alertas Recientes")
                            for alerta in st.session_state.alertas[-5:]:
                                tiempo = time.strftime('%H:%M:%S', time.localtime(alerta['timestamp']))
                                st.warning(f"**{tiempo}** - {alerta['mensaje']}")
                
                # Pequeña pausa
                time.sleep(0.03)
                
                if not st.session_state.camara_activa:
                    break
        else:
            st.info("👆 Presiona 'Iniciar Cámara' para comenzar")
            
            st.markdown("---")
            st.subheader("🎯 INSTRUCCIONES DE CALIBRACIÓN")
            st.success("""
            **PROBLEMA IDENTIFICADO:** 
            - Reporta 1.60m pero estás a 0.50m
            
            **SOLUCIÓN RÁPIDA:**
            1. Inicia la cámara
            2. Colócate a 50cm 
            3. Presiona **🎯 AUTO-CALIBRAR (50cm)**
            4. El factor se ajustará automáticamente a ~0.3
            
            **SI PERSISTE:**
            - Usa el slider para ajustar manualmente a **0.3-0.4**
            - O usa los botones ➖/➕ para ajuste fino
            """)
    
    with col2:
        st.subheader("📊 INFORMACIÓN EN TIEMPO REAL")
        
        # Mostrar detecciones actuales
        if st.session_state.sistema.detecciones_actuales:
            st.success(f"👥 **Personas detectadas:** {len(st.session_state.sistema.detecciones_actuales)}")
            
            for i, det in enumerate(st.session_state.sistema.detecciones_actuales):
                with st.expander(f"Persona {i+1} - {det['distancia_estimada']:.2f}m"):
                    st.write(f"**Detector:** {det['detector']}")
                    st.write(f"**Distancia:** {det['distancia_estimada']:.2f}m")
                    st.write(f"**Zona:** {st.session_state.sistema.determinar_zona(det['centro'][0])}")
                    
                    ancho = det['bbox'][2] - det['bbox'][0]
                    alto = det['bbox'][3] - det['bbox'][1]
                    st.write(f"**Ancho:** {ancho}px")
                    st.write(f"**Alto:** {alto}px")
                    
                    # Información de calibración
                    st.write("---")
                    st.write("**Calibración:**")
                    st.write(f"Ref. cerca: {st.session_state.sistema.calibracion_distancia['ancho_referencia_cerca']}px")
                    st.write(f"Ref. lejos: {st.session_state.sistema.calibracion_distancia['ancho_referencia_lejos']}px")
        else:
            st.info("👀 Esperando detecciones...")
        
        # Estadísticas
        st.subheader("📈 ESTADÍSTICAS")
        if st.session_state.alertas:
            st.metric("Alertas Totales", len(st.session_state.alertas))
            if st.session_state.alertas:
                ultima = st.session_state.alertas[-1]
                st.metric("Última Distancia", f"{ultima['distancia']:.2f}m")
                st.metric("Zona", ultima['zona'])
        else:
            st.info("No hay alertas registradas")
        
        # Información de calibración actual
        st.subheader("🔧 CALIBRACIÓN ACTUAL")
        cal = st.session_state.sistema.calibracion_distancia
        st.write(f"**Factor:** {cal['factor_ajuste_camara']:.3f}")
        st.write(f"**Ref. Cerca:** {cal['ancho_referencia_cerca']}px = {cal['distancia_referencia_cerca']}m")
        st.write(f"**Ref. Lejos:** {cal['ancho_referencia_lejos']}px = {cal['distancia_referencia_lejos']}m")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Sistema de Detección de Personas** | "
        "🔥 **CALIBRACIÓN PRECISA ACTIVADA** | "
        "Basado en tu caso: 50cm reales = 1.60m reportados"
    )

if __name__ == "__main__":
    main()
