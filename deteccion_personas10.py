# deteccion_personas_corregido_distancia.py
import cv2
import numpy as np
import time
import pygame
import os

class SistemaDeteccionPersonas:
    def __init__(self):
        self.ventana_creada = False
        self.inicializar_camara()
        self.inicializar_deteccion_avanzada()
        self.inicializar_audio_mejorado()
        self.configurar_parametros()
        print("✅ Sistema inicializado correctamente")
        
    def inicializar_camara(self):
        """Configura la cámara para captura de video"""
        print("📷 Inicializando cámara...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ No se pudo abrir la cámara. Intentando con índice 1...")
            self.cap = cv2.VideoCapture(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        print("✅ Cámara configurada")
        
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
            pygame.init()
            pygame.mixer.init(
                frequency=22050,
                size=-16,
                channels=2,
                buffer=512
            )
            
            if not pygame.mixer.get_init():
                print("❌ No se pudo inicializar el mixer de pygame")
                return
                
            self.audio_disponible = True
            print(f"✅ Audio inicializado: {pygame.mixer.get_init()}")
            
            self.crear_sonidos_mejorados()
            
        except Exception as e:
            print(f"❌ Error crítico inicializando audio: {e}")
            self.audio_disponible = False
            
        self.ultima_alerta = 0
        self.cooldown = 2.0
        
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
        
        # 🔥 MEJORADO: Parámetros de calibración de distancia
        self.calibracion_distancia = {
            'ancho_referencia_cerca': 250,    # Ancho en píxeles a 40cm
            'ancho_referencia_lejos': 80,     # Ancho en píxeles a 2m
            'distancia_referencia_cerca': 0.4, # 40cm
            'distancia_referencia_lejos': 2.0, # 2m
            'factor_ajuste_camara': 1.2       # Factor de ajuste para tu cámara
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
                'distancia_estimada': self.estimar_distancia_mejorada(w, h),
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
                        'distancia_estimada': self.estimar_distancia_mejorada(w, h),
                        'detector': 'movimiento'
                    })
        
        self.frame_anterior = gray
        return resultados
    
    def estimar_distancia_mejorada(self, ancho_persona, alto_persona):
        """🔥 MEJORADO: Estimación de distancia más precisa usando interpolación lineal"""
        cal = self.calibracion_distancia
        
        # Usar tanto ancho como alto para mejor precisión
        dimension_promedio = (ancho_persona + alto_persona * 0.6) / 1.6
        
        if dimension_promedio <= 0:
            return 10.0
        
        # Interpolación lineal entre puntos de calibración
        if dimension_promedio >= cal['ancho_referencia_cerca']:
            # Muy cerca - usar extrapolación
            ratio = cal['ancho_referencia_cerca'] / dimension_promedio
            distancia = cal['distancia_referencia_cerca'] * ratio
        elif dimension_promedio <= cal['ancho_referencia_lejos']:
            # Muy lejos - usar extrapolación
            ratio = cal['ancho_referencia_lejos'] / dimension_promedio
            distancia = cal['distancia_referencia_lejos'] * ratio
        else:
            # Interpolación lineal entre puntos conocidos
            rango_ancho = cal['ancho_referencia_cerca'] - cal['ancho_referencia_lejos']
            rango_distancia = cal['distancia_referencia_lejos'] - cal['distancia_referencia_cerca']
            
            proporcion = (dimension_promedio - cal['ancho_referencia_lejos']) / rango_ancho
            distancia = cal['distancia_referencia_cerca'] + (rango_distancia * proporcion)
        
        # Aplicar factor de ajuste de cámara
        distancia *= cal['factor_ajuste_camara']
        
        return max(0.2, min(15.0, distancia))
    
    def calibrar_distancia_manual(self, distancia_real, ancho_persona):
        """🔥 NUEVO: Función para calibrar manualmente el sistema"""
        print(f"🔧 Calibración: {distancia_real}m -> {ancho_persona}px")
        
        if distancia_real < 1.0:
            self.calibracion_distancia['ancho_referencia_cerca'] = ancho_persona
            self.calibracion_distancia['distancia_referencia_cerca'] = distancia_real
            print(f"✅ Punto cercano calibrado: {ancho_persona}px = {distancia_real}m")
        else:
            self.calibracion_distancia['ancho_referencia_lejos'] = ancho_persona
            self.calibracion_distancia['distancia_referencia_lejos'] = distancia_real
            print(f"✅ Punto lejano calibrado: {ancho_persona}px = {distancia_real}m")
    
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
        
        if distancia < 5.0:
            if self.audio_disponible and self.sonidos:
                try:
                    if distancia < 1.0:  # 🔥 AJUSTADO: 1m para alerta urgente
                        for _ in range(2):
                            if zona in self.sonidos:
                                self.sonidos[zona].play()
                                pygame.time.wait(200)
                        mensaje = f"🚨 MUY CERCA! {distancia:.1f}m - {zona.upper()}"
                    else:
                        if zona in self.sonidos:
                            self.sonidos[zona].play()
                        mensaje = f"🔊 Persona: {distancia:.1f}m - {zona}"
                except Exception as e:
                    print(f"⚠️  Error reproduciendo audio: {e}")
                    mensaje = f"🔇 [AUDIO ERROR] Persona: {distancia:.1f}m - {zona}"
            else:
                if distancia < 1.0:
                    mensaje = f"🚨 ALERTA: Persona MUY CERCA a {distancia:.1f}m - {zona.upper()}"
                else:
                    mensaje = f"🔊 Persona detectada: {distancia:.1f}m - {zona}"
            
            print(f"👤 {mensaje} | Detector: {persona_cercana['detector']}")
            self.ultima_alerta = tiempo_actual
    
    def dibujar_interfaz(self, frame, detecciones):
        """Dibuja interfaz mejorada con información de calibración"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.ancho_pantalla, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.line(frame, (int(self.ancho_pantalla * 0.4), 0), 
                (int(self.ancho_pantalla * 0.4), self.alto_pantalla), (255, 255, 255), 2)
        cv2.line(frame, (int(self.ancho_pantalla * 0.6), 0), 
                (int(self.ancho_pantalla * 0.6), self.alto_pantalla), (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if detecciones else (0, 0, 255)
        status_text = f"Personas: {len(detecciones)} | Audio: {'ON' if self.audio_disponible else 'OFF'}"
        cv2.putText(frame, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 🔥 NUEVO: Mostrar información de calibración
        cal_info = f"Calib: {self.calibracion_distancia['factor_ajuste_camara']:.1f}"
        cv2.putText(frame, cal_info, (self.ancho_pantalla - 100, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, det in enumerate(detecciones):
            x1, y1, x2, y2 = det['bbox']
            distancia = det['distancia_estimada']
            zona = self.determinar_zona(det['centro'][0])
            detector = det['detector']
            ancho = x2 - x1
            alto = y2 - y1
            
            if distancia < 1.0:
                color = (0, 0, 255)
            elif distancia < 2.0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{i+1}: {distancia:.1f}m ({zona})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.circle(frame, det['centro'], 5, color, -1)
            
            # 🔥 NUEVO: Mostrar dimensiones para calibración
            dim_text = f"{ancho}x{alto}"
            cv2.putText(frame, dim_text, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            cv2.putText(frame, detector, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return frame
    
    def ejecutar(self):
        """Función principal con calibración mejorada"""
        print("\n" + "="*60)
        print("🎯 SISTEMA DE DETECCIÓN - DISTANCIA MEJORADA")
        print("="*60)
        print("💡 CONSEJOS PARA CALIBRACIÓN:")
        print("   - Para calibrar: Colócate a 40cm y presiona '1'")
        print("   - Para calibrar: Colócate a 2m y presiona '2'")
        print("   - Ajustar factor: '+' para aumentar, '-' para disminuir")
        print("   - 'a': Probar audio | 'd': Debug | 'q': Salir")
        print("="*60)
        print(f"🔊 Estado audio: {'DISPONIBLE' if self.audio_disponible else 'NO DISPONIBLE'}")
        print("="*60)
        
        nombre_ventana = 'Sistema Detección - MEJORADO (Calibrar con 1/2)'
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, 800, 600)
        
        try:
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error con la cámara")
                    break
                
                frame = cv2.flip(frame, 1)
                
                if frame_count % 2 == 0:
                    detecciones = self.detectar_personas(frame)
                
                self.generar_alerta(detecciones)
                frame_ui = self.dibujar_interfaz(frame, detecciones)
                
                cv2.imshow(nombre_ventana, frame_ui)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    print("⏸️  Pausa. Presiona cualquier tecla...")
                    cv2.waitKey(0)
                elif key == ord('d'):
                    print("🔍 Modo debug activado")
                    for i, det in enumerate(detecciones):
                        ancho = det['bbox'][2] - det['bbox'][0]
                        alto = det['bbox'][3] - det['bbox'][1]
                        print(f"  {i+1}. {det['detector']} - {det['distancia_estimada']:.2f}m - {ancho}x{alto}px")
                elif key == ord('a'):
                    print("🔊 Probando sonidos...")
                    for zona in ['izquierda', 'frente', 'derecha']:
                        if zona in self.sonidos:
                            print(f"  Reproduciendo sonido {zona}...")
                            self.sonidos[zona].play()
                            pygame.time.wait(500)
                
                # 🔥 NUEVO: Comandos de calibración
                elif key == ord('1'):
                    # Calibrar a 40cm
                    if detecciones:
                        persona = min(detecciones, key=lambda x: x['distancia_estimada'])
                        ancho = persona['bbox'][2] - persona['bbox'][0]
                        self.calibrar_distancia_manual(0.4, ancho)
                    else:
                        print("❌ No hay detecciones para calibrar")
                
                elif key == ord('2'):
                    # Calibrar a 2m
                    if detecciones:
                        persona = min(detecciones, key=lambda x: x['distancia_estimada'])
                        ancho = persona['bbox'][2] - persona['bbox'][0]
                        self.calibrar_distancia_manual(2.0, ancho)
                    else:
                        print("❌ No hay detecciones para calibrar")
                
                elif key == ord('+'):
                    # Aumentar factor de ajuste
                    self.calibracion_distancia['factor_ajuste_camara'] += 0.1
                    print(f"🔧 Factor ajuste aumentado: {self.calibracion_distancia['factor_ajuste_camara']:.1f}")
                
                elif key == ord('-'):
                    # Disminuir factor de ajuste
                    self.calibracion_distancia['factor_ajuste_camara'] = max(0.5, 
                        self.calibracion_distancia['factor_ajuste_camara'] - 0.1)
                    print(f"🔧 Factor ajuste disminuido: {self.calibracion_distancia['factor_ajuste_camara']:.1f}")
                
                if frame_count % 45 == 0 and detecciones:
                    closest = min(detecciones, key=lambda x: x['distancia_estimada'])
                    ancho = closest['bbox'][2] - closest['bbox'][0]
                    alto = closest['bbox'][3] - closest['bbox'][1]
                    print(f"📊 Frame {frame_count}: {len(detecciones)} personas | Más cerca: {closest['distancia_estimada']:.2f}m ({ancho}x{alto}px)")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Programa interrumpido")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.liberar_recursos()
    
    def liberar_recursos(self):
        """Liberar recursos correctamente"""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.audio_disponible:
            pygame.mixer.quit()
            pygame.quit()
        print("✅ Sistema finalizado")

if __name__ == "__main__":
    print("Iniciando sistema mejorado de detección con calibración...")
    detector = SistemaDeteccionPersonas()
    detector.ejecutar()
