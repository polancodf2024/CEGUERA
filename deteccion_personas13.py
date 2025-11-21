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
            if 'archivo_subido' in locals() and archivo_subido is not None:
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

if __name__ == "__main__":
    main()
