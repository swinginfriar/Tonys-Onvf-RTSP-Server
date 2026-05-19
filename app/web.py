import json
import os
import sys
from urllib.parse import quote
try:
    import psutil
except ImportError:
    psutil = None
import time
import functools
from datetime import timedelta
from flask import Flask, jsonify, request, session, redirect, url_for, make_response, send_file
from flask_cors import CORS

from .web_template import get_web_ui_html
from .diagnostics_template import get_diagnostics_html
from .ip_management_template import get_ip_management_html

from .ffmpeg_manager import FFmpegManager
from .onvif_client import ONVIFProber
from .linux_network import LinuxNetworkManager
from .utils import get_captured_logs
from .updater import UpdateChecker, check_for_updates, download_and_apply_update
import subprocess
import threading
import tempfile
import shutil


def create_web_app(manager):
    """Create Flask web application"""
    app = Flask(__name__)
    CORS(app)
    
    # Session configuration
    app.secret_key = getattr(manager, 'secret_key', os.urandom(24))
    app.permanent_session_lifetime = timedelta(days=30)
    
    # Initialize stats tracking
    app.stats_last_time = time.time()
    app.stats_last_cpu = 0
    
    import logging
    log = logging.getLogger('werkzeug')
    if getattr(manager, 'debug_mode', False):
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.ERROR)

    # --- Authentication Decorator ---
    def login_required(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not manager.auth_enabled:
                return f(*args, **kwargs)
                
            if 'authenticated' not in session:
                if request.is_json:
                    return jsonify({'error': 'Authentication required'}), 401
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function

    # --- Auth Routes ---
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if manager.is_setup_required():
            return redirect(url_for('setup'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = request.form.get('remember') == 'true'
            
            if manager.verify_login(username, password):
                session.permanent = remember
                session['authenticated'] = True
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
            
        from .web_template import get_login_html
        return get_login_html()

    @app.route('/setup', methods=['GET', 'POST'])
    def setup():
        if not manager.is_setup_required():
            return redirect(url_for('login'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                return jsonify({'success': False, 'error': 'Username and password required'}), 400
                
            manager.setup_user(username, password)
            session.permanent = True
            session['authenticated'] = True
            return jsonify({'success': True})
            
        from .web_template import get_setup_html
        return get_setup_html()

    @app.route('/setup/skip', methods=['POST'])
    def skip_setup():
        if not manager.is_setup_required():
            return jsonify({'success': False, 'error': 'Setup already completed'}), 400
            
        manager.skip_setup()
        session['authenticated'] = True # Mark as "logged in" for this session
        return jsonify({'success': True})

    @app.route('/logout')
    def logout():
        session.pop('authenticated', None)
        return redirect(url_for('login'))

    @app.route('/api/onvif/probe', methods=['POST'])
    @login_required
    def probe_onvif():
        """Probe an ONVIF camera for profiles"""
        data = request.json
        host = data.get('host')
        port = int(data.get('port', 80))
        username = data.get('username')
        password = data.get('password')
        
        if not host or not username or not password:
            return jsonify({'error': 'Host, username, and password are required'}), 400
            
        prober = ONVIFProber()
        result = prober.probe(host, port, username, password)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/server/restart', methods=['POST'])
    @login_required
    def restart_server():
        """Restart the RTSP server"""
        def do_restart():
            import time
            time.sleep(2)  # Give time for response to be sent
            print("\n\nServer restart requested from web UI...")
            
            # Check if running on Linux
            if sys.platform.startswith('linux'):
                print("Linux detected - performing full server restart...")
                # Stop everything cleanly
                print("Stopping MediaMTX...")
                manager.mediamtx.stop()
                print("Stopping all cameras...")
                for camera in manager.cameras:
                    camera.stop()
                
                print("Killing server process...")
                # Exit with special code 42 to signal restart needed
                # This immediately releases all ports
                os._exit(42)
            else:
                # Windows - just restart MediaMTX
                print("Stopping MediaMTX...")
                manager.mediamtx.stop()
                print("Restarting MediaMTX...")
                # Use global credentials if RTSP auth is enabled
                rtsp_user = manager.global_username if getattr(manager, 'rtsp_auth_enabled', False) else ''
                rtsp_pass = manager.global_password if getattr(manager, 'rtsp_auth_enabled', False) else ''
                manager.mediamtx.start(manager.cameras, manager.rtsp_port, rtsp_user, rtsp_pass)
                print("Server restarted successfully!\n")
            
        # Run restart in background thread
        import threading
        restart_thread = threading.Thread(target=do_restart, daemon=True)
        restart_thread.start()
        
        return jsonify({'message': 'Server restart initiated'})

    @app.route('/api/server/reboot', methods=['POST'])
    @login_required
    def reboot_server():
        """Reboot the entire server (Linux only)"""
        # Only allow on Linux
        if not sys.platform.startswith('linux'):
            return jsonify({'error': 'Reboot is only supported on Linux'}), 400
        
        def do_reboot():
            import time
            time.sleep(2)  # Give time for response to be sent
            print("\n\nServer reboot requested from web UI...")
            print("Stopping MediaMTX...")
            manager.mediamtx.stop()
            print("Stopping all cameras...")
            for camera in manager.cameras:
                camera.stop()
            
            print("Initiating system reboot...")
            # Execute system reboot command
            subprocess.run(['sudo', 'reboot'], check=False)
            
        # Run reboot in background thread
        import threading
        reboot_thread = threading.Thread(target=do_reboot, daemon=True)
        reboot_thread.start()
        
        return jsonify({'message': 'Server reboot initiated'})


    @app.route('/api/stats')
    def get_stats():
        """Get CPU and memory usage for the app and its children using delta timings"""
        if psutil is None:
            return jsonify({'cpu_percent': 0.0, 'memory_mb': 0.0})

        try:
            current_time = time.time()
            parent = psutil.Process(os.getpid())
            
            # Memory (snapshot)
            memory_info = parent.memory_info().rss
            # CPU Times (cumulative)
            total_cpu_time = parent.cpu_times().user + parent.cpu_times().system
            
            # Sum up all children recursively
            for child in parent.children(recursive=True):
                try:
                    memory_info += child.memory_info().rss
                    total_cpu_time += child.cpu_times().user + child.cpu_times().system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Calculate delta since last request
            delta_time = current_time - app.stats_last_time
            delta_cpu = total_cpu_time - app.stats_last_cpu
            
            # Update baseline for next request
            app.stats_last_time = current_time
            app.stats_last_cpu = total_cpu_time
            
            # Normalization
            cpu_count = psutil.cpu_count() or 1
            if delta_time > 0:
                # percentage = (seconds_of_cpu / seconds_of_wallclock) * 100
                # Divided by cores to get 0-100% total system view
                cpu_percent = (delta_cpu / delta_time) * 100 / cpu_count
            else:
                cpu_percent = 0.0
            
            return jsonify({
                'cpu_percent': min(100.0, round(max(0.0, cpu_percent), 1)),
                'memory_mb': round(memory_info / (1024 * 1024), 1)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics')
    @login_required
    def get_analytics():
        """Get per-stream analytics from MediaMTX"""
        try:
            return jsonify(manager.analytics.get_analytics())
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    @app.route('/')
    @login_required
    def index():
        settings = manager.load_settings()
        response = app.make_response(get_web_ui_html(settings))
        # Add headers to prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    @app.route('/gridfusion')
    @login_required
    def gridfusion():
        settings = manager.load_settings()
        grid_fusion_config = manager.get_grid_fusion()
        from .gridfusion_template import get_gridfusion_html
        return get_gridfusion_html(settings, grid_fusion_config)
    
    @app.route('/api/cameras', methods=['GET'])
    @login_required
    def get_cameras():
        return jsonify([cam.to_dict() for cam in manager.cameras])
    
    @app.route('/api/cameras', methods=['POST'])
    @login_required
    def add_camera():
        data = request.json
        try:
            camera = manager.add_camera(
                name=data['name'],
                host=data['host'],
                rtsp_port=data['rtspPort'],
                username=data.get('username', ''),
                password=data.get('password', ''),
                main_path=data['mainPath'],
                sub_path=data['subPath'],
                auto_start=data.get('autoStart', False),
                main_width=data.get('mainWidth', 1920),
                main_height=data.get('mainHeight', 1080),
                sub_width=data.get('subWidth', 640),
                sub_height=data.get('subHeight', 480),
                main_framerate=data.get('mainFramerate', 30),
                sub_framerate=data.get('subFramerate', 15),
                onvif_port=data.get('onvifPort'),
                transcode_sub=data.get('transcodeSub', False),
                transcode_main=data.get('transcodeMain', False),
                disable_substream=data.get('disableSubstream', False),
                use_main_as_substream=data.get('useMainAsSubstream', False),
                enable_audio=data.get('enableAudio', False),
                transcode_main_audio=data.get('transcodeMainAudio', False),
                transcode_sub_audio=data.get('transcodeSubAudio', False),
                use_virtual_nic=data.get('useVirtualNic', False),
                parent_interface=data.get('parentInterface', ''),
                nic_mac=data.get('nicMac', ''),
                ip_mode=data.get('ipMode', 'dhcp'),
                static_ip=data.get('staticIp', ''),
                netmask=data.get('netmask', '24'),
                gateway=data.get('gateway', ''),
                uuid=data.get('uuid')
            )
            return jsonify(camera.to_dict()), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/cameras/<int:camera_id>', methods=['PUT'])
    @login_required
    def update_camera(camera_id):
        data = request.json
        try:
            camera = manager.update_camera(
                camera_id=camera_id,
                name=data['name'],
                host=data['host'],
                rtsp_port=data['rtspPort'],
                username=data.get('username', ''),
                password=data.get('password', ''),
                main_path=data['mainPath'],
                sub_path=data['subPath'],
                auto_start=data.get('autoStart', False),
                main_width=data.get('mainWidth', 1920),
                main_height=data.get('mainHeight', 1080),
                sub_width=data.get('subWidth', 640),
                sub_height=data.get('subHeight', 480),
                main_framerate=data.get('mainFramerate', 30),
                sub_framerate=data.get('subFramerate', 15),
                onvif_port=data.get('onvifPort'),
                transcode_sub=data.get('transcodeSub', False),
                transcode_main=data.get('transcodeMain', False),
                disable_substream=data.get('disableSubstream', False),
                use_main_as_substream=data.get('useMainAsSubstream', False),
                enable_audio=data.get('enableAudio', False),
                transcode_main_audio=data.get('transcodeMainAudio', False),
                transcode_sub_audio=data.get('transcodeSubAudio', False),
                use_virtual_nic=data.get('useVirtualNic', False),
                parent_interface=data.get('parentInterface', ''),
                nic_mac=data.get('nicMac', ''),
                ip_mode=data.get('ipMode', 'dhcp'),
                static_ip=data.get('staticIp', ''),
                netmask=data.get('netmask', '24'),
                gateway=data.get('gateway', ''),
                uuid=data.get('uuid')
            )
            if camera:
                return jsonify(camera.to_dict())
            return jsonify({'error': 'Camera not found'}), 404
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
    @login_required
    def delete_camera(camera_id):
        if manager.delete_camera(camera_id):
            return '', 204
        return jsonify({'error': 'Camera not found'}), 404
    
    @app.route('/api/cameras/<int:camera_id>/start', methods=['POST'])
    @login_required
    def start_camera(camera_id):
        camera = manager.get_camera(camera_id)
        if camera:
            # Only restart MediaMTX if camera wasn't already running
            was_running = camera.status == "running"
            camera.start()
            manager.save_config()
            if not was_running:
                rtsp_user = manager.global_username if getattr(manager, 'rtsp_auth_enabled', False) else ''
                rtsp_pass = manager.global_password if getattr(manager, 'rtsp_auth_enabled', False) else ''
                manager.mediamtx.restart(manager.cameras, manager.rtsp_port, rtsp_user, rtsp_pass, manager.get_grid_fusion())
            return jsonify(camera.to_dict())
        return jsonify({'error': 'Camera not found'}), 404
    
    @app.route('/api/cameras/<int:camera_id>/stop', methods=['POST'])
    @login_required
    def stop_camera(camera_id):
        camera = manager.get_camera(camera_id)
        if camera:
            # Only restart MediaMTX if camera was actually running
            was_running = camera.status == "running"
            camera.stop()
            manager.save_config()
            if was_running:
                rtsp_user = manager.global_username if getattr(manager, 'rtsp_auth_enabled', False) else ''
                rtsp_pass = manager.global_password if getattr(manager, 'rtsp_auth_enabled', False) else ''
                manager.mediamtx.restart(manager.cameras, manager.rtsp_port, rtsp_user, rtsp_pass, manager.get_grid_fusion())
            return jsonify(camera.to_dict())
        return jsonify({'error': 'Camera not found'}), 404

    @app.route('/api/cameras/<int:camera_id>/motion/start', methods=['POST'])
    @login_required
    def trigger_motion_start(camera_id):
        """Synthetic motion trigger: emit ONVIF motion-on event for this camera.

        Lets users validate the NVR's ONVIF subscription/event ingestion without
        depending on a working motion detector. Also used by automated tests.
        """
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        if camera.status != "running":
            return jsonify({'error': 'Camera is not running'}), 400
        from .motion_controller import get_motion_controller
        state = get_motion_controller().set_motion(camera, True, source='synthetic', immediate=True)
        return jsonify({'cameraId': camera.id, 'motion': state, 'source': 'synthetic'})

    @app.route('/api/cameras/<int:camera_id>/motion/stop', methods=['POST'])
    @login_required
    def trigger_motion_stop(camera_id):
        """Synthetic motion clear: emit ONVIF motion-off event for this camera."""
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        if camera.status != "running":
            return jsonify({'error': 'Camera is not running'}), 400
        from .motion_controller import get_motion_controller
        state = get_motion_controller().set_motion(camera, False, source='synthetic', immediate=True)
        return jsonify({'cameraId': camera.id, 'motion': state, 'source': 'synthetic'})

    @app.route('/api/cameras/<int:camera_id>/motion/state', methods=['GET'])
    @login_required
    def motion_state(camera_id):
        """Report the current debounced motion state for this camera."""
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        from .motion_controller import get_motion_controller
        return jsonify({'cameraId': camera.id, 'motion': get_motion_controller().get_state(camera.id)})

    @app.route('/api/classifier/models', methods=['GET'])
    @login_required
    def classifier_models():
        """Return available classification models + whether ultralytics is installed.

        The UI uses `installed` to decide whether to show an install prompt
        instead of letting the user enable classification on a broken setup.
        """
        from . import classifier
        return jsonify({
            'installed': classifier.is_available(),
            'install_command': 'pip install ultralytics',
            'default_model': classifier.DEFAULT_MODEL,
            'models': classifier.list_models(),
        })

    @app.route('/api/classifier/classes', methods=['GET'])
    @login_required
    def classifier_classes():
        """Return the grouped COCO class catalog for the UI's checkbox UI."""
        from . import classifier
        return jsonify({
            'default_classes': classifier.DEFAULT_CLASSES,
            'groups': classifier.list_classes(),
        })

    @app.route('/api/cameras/<int:camera_id>/snapshot', methods=['GET'])
    @login_required
    def admin_snapshot(camera_id):
        """Return a JPEG snapshot of the camera for the admin UI (zone editor).

        Server-side proxy of the camera's substream so the browser doesn't
        need to hit the per-camera ONVIF port with credentials. Mirrors the
        approach used by /onvif/snapshot but skips ONVIF auth (login_required
        already gates it).
        """
        import os, tempfile
        from urllib.parse import quote
        from .ffmpeg_manager import FFmpegManager

        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404

        suffix = '_sub' if not getattr(camera, 'disable_substream', False) else '_main'
        rtsp_port = getattr(manager, 'rtsp_port', 8554)
        if getattr(manager, 'rtsp_auth_enabled', False):
            user = quote(getattr(manager, 'global_username', 'admin') or 'admin', safe='')
            pwd = quote(getattr(manager, 'global_password', 'admin') or 'admin', safe='')
            url = f"rtsp://{user}:{pwd}@127.0.0.1:{rtsp_port}/{camera.path_name}{suffix}"
        else:
            url = f"rtsp://127.0.0.1:{rtsp_port}/{camera.path_name}{suffix}"

        fd, path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            ok, err = FFmpegManager().capture_snapshot(url, path)
            if not ok:
                return jsonify({'error': f'Snapshot failed: {err}'}), 502
            with open(path, 'rb') as f:
                content = f.read()
            from flask import make_response
            resp = make_response(content)
            resp.headers['Content-Type'] = 'image/jpeg'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    @app.route('/api/cameras/<int:camera_id>/motion/config', methods=['GET'])
    @login_required
    def motion_config_get(camera_id):
        """Return this camera's motion detection config."""
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        return jsonify({'cameraId': camera.id, 'motion': getattr(camera, 'motion', {}) or {}})

    @app.route('/api/cameras/<int:camera_id>/motion/config', methods=['PUT'])
    @login_required
    def motion_config_put(camera_id):
        """Update this camera's motion detection config.

        Merges incoming fields into the existing config (so partial updates
        are fine), persists to disk, and reconciles the motion worker
        (start/stop/restart) so changes take effect without restarting the
        whole camera. Returns the resulting config.
        """
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({'error': 'Body must be a JSON object'}), 400

        existing = dict(getattr(camera, 'motion', {}) or {})
        # Type-coerce the keys we know about so the UI can send strings safely
        coercions = {
            'enabled': lambda v: bool(v),
            'fps': lambda v: max(1, int(v)),
            'min_area_percent': lambda v: max(0.01, float(v)),
            'min_motion_frames': lambda v: max(1, int(v)),
            'alarm_on_delay_ms': lambda v: max(0, int(v)),
            'alarm_off_delay_ms': lambda v: max(0, int(v)),
            'min_event_duration_ms': lambda v: max(0, int(v)),
            'pre_buffer_ms': lambda v: max(0, int(v)),
        }
        for k, coerce in coercions.items():
            if k in body:
                try:
                    existing[k] = coerce(body[k])
                except (TypeError, ValueError) as e:
                    return jsonify({'error': f'Invalid value for {k}: {e}'}), 400
        # Zones are passed through as-is; MotionWorker validates per-zone.
        if 'zones' in body:
            if not isinstance(body['zones'], list):
                return jsonify({'error': 'zones must be a list'}), 400
            existing['zones'] = body['zones']

        # Classification sub-config (optional). Merge field-by-field so the
        # UI can send partial updates without clobbering unset fields.
        if 'classification' in body:
            cls_in = body['classification']
            if not isinstance(cls_in, dict):
                return jsonify({'error': 'classification must be an object'}), 400
            cls_existing = dict(existing.get('classification') or {})
            cls_coercions = {
                'enabled': lambda v: bool(v),
                'model': lambda v: str(v),
                'min_confidence': lambda v: max(0.0, min(1.0, float(v))),
                'stream': lambda v: 'main' if str(v).lower() == 'main' else 'sub',
            }
            for k, coerce in cls_coercions.items():
                if k in cls_in:
                    try:
                        cls_existing[k] = coerce(cls_in[k])
                    except (TypeError, ValueError) as e:
                        return jsonify({'error': f'Invalid classification.{k}: {e}'}), 400
            if 'classes' in cls_in:
                if not isinstance(cls_in['classes'], list):
                    return jsonify({'error': 'classification.classes must be a list'}), 400
                cls_existing['classes'] = [str(c) for c in cls_in['classes']]
            existing['classification'] = cls_existing

        camera.motion = existing
        manager.save_config()

        # Reconcile the worker — start/stop/restart depending on new enabled state.
        from . import motion_detector
        motion_detector.start_worker(camera)

        return jsonify({'cameraId': camera.id, 'motion': camera.motion})

    @app.route('/api/cameras/start-all', methods=['POST'])
    @login_required
    def start_all():
        manager.start_all()
        return jsonify([cam.to_dict() for cam in manager.cameras])
    
    @app.route('/api/cameras/stop-all', methods=['POST'])
    @login_required
    def stop_all():
        manager.stop_all()
        return jsonify([cam.to_dict() for cam in manager.cameras])
    
    @app.route('/api/cameras/reset-uuids', methods=['POST'])
    @login_required
    def reset_uuids():
        manager.reset_all_uuids()
        return jsonify({'status': 'success', 'message': 'All camera UUIDs have been reset.'})

    @app.route('/api/cameras/reset-macs', methods=['POST'])
    @login_required
    def reset_macs():
        manager.reset_all_macs()
        return jsonify({'status': 'success', 'message': 'All camera MAC addresses have been reset.'})

    @app.route('/api/cameras/<int:camera_id>/fetch-stream-info', methods=['POST'])
    @login_required
    def fetch_stream_info(camera_id):
        """Fetch stream information using FFprobe"""
        data = request.json
        stream_type = data.get('streamType', 'main')  # 'main' or 'sub'
        
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        
        # Get the appropriate stream URL
        stream_url = camera.main_stream_url if stream_type == 'main' else camera.sub_stream_url
        
        try:
            # Use ffprobe to get stream information
            
            # Get ffprobe path (will download if needed)
            ffmpeg_manager = FFmpegManager()
            ffprobe_path = ffmpeg_manager.get_ffprobe_path()
            
            if not ffprobe_path:
                return jsonify({
                    'error': 'FFprobe not available and could not be downloaded automatically.',
                    'installUrl': 'https://ffmpeg.org/download.html'
                }), 400
            
            print(f"  Using ffprobe: {ffprobe_path}")
            print(f"  Probing stream: {stream_url}")
            
            # Run ffprobe to get stream info
            # Use TCP for better compatibility with cameras
            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-rtsp_transport', 'tcp',  # Use TCP instead of UDP for better compatibility
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'json',
                stream_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                # Log the error for debugging
                print(f"  FFprobe failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
                print(f"  stdout: {result.stdout}")
                
                # Provide helpful error messages based on common issues
                error_msg = 'Failed to probe stream.'
                troubleshooting = []
                
                if '5XX Server Error' in result.stderr:
                    troubleshooting.append('• Camera connection limit might be reached (too many concurrent streams)')
                    troubleshooting.append('• Reboot the camera if it is unresponsive')
                    troubleshooting.append('• Verify the stream path/URL is correct')
                elif '401' in result.stderr or '403' in result.stderr:
                    troubleshooting.append('• Check camera credentials (username/password)')
                    troubleshooting.append('• verify the stream path is correct')
                elif 'Connection refused' in result.stderr or 'Connection timed out' in result.stderr:
                    troubleshooting.append('• Check if camera IP address is correct')
                    troubleshooting.append('• Verify camera is powered on and accessible')
                    troubleshooting.append('• Check network connectivity')
                elif 'Invalid data found' in result.stderr:
                    troubleshooting.append('• Stream path might be incorrect')
                    troubleshooting.append('• Camera might not be streaming on this path')
                else:
                    troubleshooting.append('• Verify stream URL is accessible')
                    troubleshooting.append('• Check camera is not overloaded with connections')
                    troubleshooting.append('• Try accessing the stream in VLC to confirm it works')
                
                return jsonify({
                    'error': error_msg,
                    'details': result.stderr,
                    'troubleshooting': troubleshooting,
                    'returnCode': result.returncode
                }), 400
            
            # Parse the JSON output
            import json as json_module
            probe_data = json_module.loads(result.stdout)
            
            if 'streams' not in probe_data or len(probe_data['streams']) == 0:
                return jsonify({'error': 'No video stream found'}), 400
            
            stream_info = probe_data['streams'][0]
            width = stream_info.get('width')
            height = stream_info.get('height')
            
            # Parse frame rate (format: "30/1" or "30000/1001")
            framerate = 30  # default
            r_frame_rate = stream_info.get('r_frame_rate', '30/1')
            if '/' in r_frame_rate:
                num, den = r_frame_rate.split('/')
                framerate = round(int(num) / int(den))
            
            return jsonify({
                'width': width,
                'height': height,
                'framerate': framerate,
                'streamType': stream_type
            })
            
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Stream probe timeout. Check if the camera is accessible.'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to fetch stream info: {str(e)}'}), 500
    
    @app.route('/api/cameras/<int:camera_id>/auto-start', methods=['POST'])
    @login_required
    def toggle_auto_start(camera_id):
        """Toggle auto-start setting for a camera"""
        data = request.json
        auto_start = data.get('autoStart', False)
        
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
        
        try:
            # Update auto-start setting
            camera.auto_start = auto_start
            manager.save_config()
            
            print(f"  Updated auto-start for {camera.name}: {auto_start}")
            
            return jsonify(camera.to_dict())
        except Exception as e:
            print(f"  Error updating auto-start: {e}")
            return jsonify({'error': str(e)}), 500
    

    
    @app.route('/api/server/stop', methods=['POST'])
    @login_required
    def stop_server():
        """Stop the entire server"""
        def do_stop():
            import time
            import os
            import signal
            import subprocess
            time.sleep(2)  # Give time for response to be sent
            print("\n\nServer stop requested from web UI...")
            print("Stopping MediaMTX...")
            manager.mediamtx.stop()
            print("Stopping all cameras...")
            for camera in manager.cameras:
                camera.stop()
            print("Server stopped successfully!")
            print("\nTo restart, run the script again.\n")
            
            # Check if running as systemd service
            try:
                if sys.platform.startswith('linux'):
                    # Check if we're running under systemd
                    result = subprocess.run(['systemctl', 'is-active', 'tonys-onvif'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0 and result.stdout.strip() == 'active':
                        print("Detected systemd service. Stopping service...")
                        # Stop the systemd service properly
                        subprocess.run(['systemctl', 'stop', 'tonys-onvif'], timeout=5)
                        return
                    else:
                        # Not running as service, kill process group
                        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                else:
                    # On Windows, just exit normally
                    os._exit(0)
            except:
                # Fallback to regular exit
                os._exit(0)
        
        # Run stop in background thread
        import threading
        stop_thread = threading.Thread(target=do_stop, daemon=True)
        stop_thread.start()
        
        return jsonify({'message': 'Server stop initiated'})
    
    @app.route('/api/settings', methods=['GET'])
    @login_required
    def get_settings():
        return jsonify(manager.load_settings())
    
    @app.route('/api/settings', methods=['POST'])
    @login_required
    def save_settings():
        data = request.json
        try:
            settings = manager.save_settings(data)
            return jsonify(settings)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/system/versions', methods=['GET'])
    @login_required
    def get_system_versions():
        """Get MediaMTX and FFmpeg versions"""
        try:
            from .ffmpeg_manager import FFmpegManager
            import shutil
            
            # Get MediaMTX version
            mediamtx_version = manager.mediamtx._get_latest_version()
            
            # Get FFmpeg version - use the same logic as the rest of the app
            ffmpeg_mgr = FFmpegManager()
            ffmpeg_version_tuple = ffmpeg_mgr.get_active_version()
            
            if ffmpeg_version_tuple:
                ffmpeg_version = f"{ffmpeg_version_tuple[0]}.{ffmpeg_version_tuple[1]}.{ffmpeg_version_tuple[2]}"
            else:
                ffmpeg_version = "Not installed"
            
            return jsonify({
                'mediamtx': mediamtx_version,
                'ffmpeg': ffmpeg_version
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/config/backup', methods=['GET'])
    @login_required
    def backup_config():
        """Download configuration backup"""
        try:
            return send_file(
                manager.config_file,
                mimetype='application/json',
                as_attachment=True,
                download_name='camera_config.json'
            )
        except Exception as e:
            return jsonify({'error': f'Failed to download config: {str(e)}'}), 500

    @app.route('/api/config/restore', methods=['POST'])
    @login_required
    def restore_config():
        """Restore configuration from backup"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            try:
                # Read and validate JSON first
                content = file.read()
                try:
                    config_data = json.loads(content)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON file'}), 400
                
                # Basic validation: Check for 'cameras' or 'settings' keys
                if 'cameras' not in config_data and 'settings' not in config_data:
                    return jsonify({'error': 'Invalid configuration file format'}), 400

                # Save file
                with open(manager.config_file, 'wb') as f:
                    f.write(content)
                
                # Reload config in manager
                manager.load_config()
                
                # Trigger server restart to apply all changes
                def do_restart():
                    time.sleep(1)
                    print("\n\nWait... Config restored. Restarting system...")
                    manager.mediamtx.stop()
                    
                    rtsp_user = manager.global_username if getattr(manager, 'rtsp_auth_enabled', False) else ''
                    rtsp_pass = manager.global_password if getattr(manager, 'rtsp_auth_enabled', False) else ''
                    manager.mediamtx.start(manager.cameras, manager.rtsp_port, rtsp_user, rtsp_pass, manager.get_grid_fusion())
                
                import threading
                threading.Thread(target=do_restart, daemon=True).start()
                
                return jsonify({'success': True, 'message': 'Configuration restored. Server restarting...'})
            except Exception as e:
                 return jsonify({'error': f'Failed to restore config: {str(e)}'}), 500

    @app.route('/api/logs', methods=['GET'])
    @login_required
    def get_logs():
        """Retrieve captured terminal logs"""
        return jsonify({'logs': get_captured_logs()})
    
    @app.route('/api/network/interfaces')
    @login_required
    def get_network_interfaces():
        """Get list of physical network interfaces (Linux only)"""
        if not LinuxNetworkManager.is_linux():
            return jsonify([])
        
        interfaces = LinuxNetworkManager.get_physical_interfaces()
        return jsonify(interfaces)
    
    # --- GridFusion Endpoints ---
    
    @app.route('/api/gridfusion', methods=['GET'])
    @login_required
    def get_gridfusion():
        """Get GridFusion configuration"""
        return jsonify(manager.get_grid_fusion())
    
    @app.route('/api/gridfusion', methods=['POST'])
    @login_required
    def save_gridfusion():
        """Save GridFusion configuration"""
        data = request.json
        try:
            result = manager.save_grid_fusion(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/gridfusion/debug', methods=['GET'])
    @login_required
    def get_gridfusion_debug():
        """Get real-time debug info for GridFusion from MediaMTX logs"""
        # Get logs from mediamtx manager buffer
        with manager.mediamtx._log_lock:
            logs = list(manager.mediamtx.log_buffer)
        
        # Look for the last 'speed=' in the logs
        speed = "unknown"
        for line in reversed(logs):
            if "speed=" in line:
                import re
                # FFmpeg speed output looks like: speed=1.01x
                match = re.search(r'speed=\s*([\d.]+x)', line)
                if match:
                    speed = match.group(1)
                    break
        
        return jsonify({
            'speed': speed,
            'log_tail': logs[-10:] if logs else []
        })

    @app.route('/api/gridfusion/snapshot/<int:camera_id>')
    @login_required
    def get_camera_snapshot(camera_id):
        """Get a single snapshot from a camera stream"""
        camera = manager.get_camera(camera_id)
        if not camera:
            return jsonify({'error': 'Camera not found'}), 404
            
        # Optimization: If camera is running, pull from local MediaMTX instead of hitting real camera
        # This is MUCH faster and avoids overloading physical cameras
        if camera.status == "running":
            rtsp_port = getattr(manager, 'rtsp_port', 8554)
            # Include credentials if RTSP auth is enabled
            if getattr(manager, 'rtsp_auth_enabled', False):
                user = quote(getattr(manager, 'global_username', 'admin'))
                pw = quote(getattr(manager, 'global_password', 'admin'))
                stream_url = f"rtsp://{user}:{pw}@localhost:{rtsp_port}/{camera.path_name}_sub"
            else:
                stream_url = f"rtsp://localhost:{rtsp_port}/{camera.path_name}_sub"
            print(f"  Capture: Using local stream for {camera.name}")
        else:
            # Fallback to direct camera URL (use sub stream for speed)
            stream_url = camera.sub_stream_url
            print(f"  Capture: Using direct stream for {camera.name}")
        
        ffmpeg_mgr = FFmpegManager()
        
        # Create a temp file for the snapshot
        fd, path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        
        try:
            success, error = ffmpeg_mgr.capture_snapshot(stream_url, path)
            
            if success:
                # Send file content
                with open(path, 'rb') as f:
                    content = f.read()
                    
                response = make_response(content)
                response.headers['Content-Type'] = 'image/jpeg'
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                return response
            else:
                print(f"  Error capturing snapshot for {camera.name}: {error}")
                return jsonify({'error': error}), 500
            
        except Exception as e:
            print(f"  Error capturing snapshot for {camera.name}: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

    
    # --- Update System Endpoints ---
    
    # Global variable to track update progress
    update_progress = {'status': 'idle', 'progress': 0, 'message': ''}
    
    @app.route('/api/updates/check', methods=['GET'])
    @login_required
    def check_updates():
        """Check for available updates from GitHub"""
        try:
            update_info = check_for_updates()
            if update_info:
                return jsonify(update_info)
            else:
                return jsonify({'error': 'Failed to check for updates'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/updates/apply', methods=['POST'])
    @login_required
    def apply_update():
        """Download and apply update"""
        data = request.json
        download_url = data.get('download_url')
        
        if not download_url:
            return jsonify({'error': 'Download URL required'}), 400
        
        def progress_callback(status, progress):
            """Update progress for frontend polling"""
            update_progress['status'] = status
            update_progress['progress'] = progress
            if status == 'downloading':
                update_progress['message'] = f'Downloading update... {int(progress)}%'
            elif status == 'backing_up':
                update_progress['message'] = 'Creating backup...'
            elif status == 'extracting':
                update_progress['message'] = 'Extracting files...'
            elif status == 'applying':
                update_progress['message'] = 'Applying update...'
            elif status == 'complete':
                update_progress['message'] = 'Update complete! Restarting server...'
        
        def do_update():
            try:
                # Reset progress
                update_progress['status'] = 'starting'
                update_progress['progress'] = 0
                update_progress['message'] = 'Initializing update...'
                
                # Download and apply update
                success = download_and_apply_update(download_url, progress_callback)
                
                if success:
                    update_progress['status'] = 'complete'
                    update_progress['progress'] = 100
                    update_progress['message'] = 'Update complete! Restarting server...'
                    
                    # Wait a moment for the status to be read
                    time.sleep(2)
                    
                    # Restart server
                    print("\n\nUpdate applied successfully! Restarting server...")
                    manager.mediamtx.stop()
                    
                    # Exit with code 42 to trigger restart (Linux) or just exit (Windows)
                    if sys.platform.startswith('linux'):
                        os._exit(42)
                    else:
                        print("\nPlease restart the server manually to complete the update.")
                        os._exit(0)
                else:
                    update_progress['status'] = 'error'
                    update_progress['message'] = 'Update failed. Check logs for details.'
            except Exception as e:
                update_progress['status'] = 'error'
                update_progress['message'] = f'Update failed: {str(e)}'
                print(f"Update error: {e}")
        
        # Start update in background thread
        import threading
        update_thread = threading.Thread(target=do_update, daemon=True)
        update_thread.start()
        
        return jsonify({'message': 'Update started', 'status': 'started'})
    
    @app.route('/api/updates/status', methods=['GET'])
    @login_required
    def get_update_status():
        """Get current update progress"""
        return jsonify(update_progress)
    
    # --- Diagnostics Endpoints ---
    
    @app.route('/diagnostics')
    @login_required
    def diagnostics():
        """Serve the diagnostic tools page"""
        return get_diagnostics_html()
        
    @app.route('/api/diagnostics/ping', methods=['POST'])
    @login_required
    def diag_ping():
        """Run ping test"""
        data = request.json
        host = data.get('host')
        count = min(10, data.get('count', 4))
        
        if not host:
            return jsonify({'success': False, 'error': 'Host required'}), 400
            
        try:
            # -n on Windows, -c on Linux
            param = '-n' if sys.platform.startswith('win') else '-c'
            cmd = ['ping', param, str(count), host]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return jsonify({
                'success': True, 
                'output': result.stdout if result.returncode == 0 else result.stderr
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/diagnostics/traceroute', methods=['POST'])
    @login_required
    def diag_traceroute():
        """Run traceroute test"""
        data = request.json
        host = data.get('host')
        
        if not host:
            return jsonify({'success': False, 'error': 'Host required'}), 400
            
        try:
            # tracert on Windows, traceroute on Linux
            cmd_name = 'tracert' if sys.platform.startswith('win') else 'traceroute'
            cmd = [cmd_name, host]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return jsonify({
                'success': True, 
                'output': result.stdout if result.returncode == 0 else result.stderr
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/diagnostics/port-check', methods=['POST'])
    @login_required
    def diag_port_check():
        """Check if a specific port is open"""
        import socket
        data = request.json
        host = data.get('host')
        port = int(data.get('port', 554))
        
        if not host:
            return jsonify({'success': False, 'error': 'Host required'}), 400
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            return jsonify({'success': True, 'open': result == 0})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/diagnostics/stream-test', methods=['POST'])
    @login_required
    def diag_stream_test():
        """Test RTSP stream with ffprobe"""
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'error': 'URL required'}), 400
            
        try:
            ffmpeg_mgr = FFmpegManager()
            ffprobe_exe = ffmpeg_mgr.get_ffprobe_path()
            
            if not ffprobe_exe:
                return jsonify({'success': False, 'error': 'FFprobe not found'}), 404
                
            cmd = [
                ffprobe_exe,
                '-v', 'error',  # Show errors but suppress warnings about missing reference frames
                '-rtsp_transport', 'tcp',
                '-analyzeduration', '5000000',  # Analyze up to 5 seconds of stream
                '-probesize', '5000000',  # Read up to 5MB to find stream info
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                # Combine stderr and stdout for better error context
                error_msg = result.stderr.strip() if result.stderr.strip() else result.stdout.strip()
                if not error_msg:
                    error_msg = f'Connection failed (exit code: {result.returncode})'
                
                # Log the full error for debugging
                print(f"  [Stream Test] FFprobe failed for URL: {url}")
                print(f"  [Stream Test] Return code: {result.returncode}")
                print(f"  [Stream Test] Stderr: {result.stderr}")
                print(f"  [Stream Test] Stdout: {result.stdout}")
                
                return jsonify({'success': False, 'error': error_msg}), 400
                
            import json as json_mod
            try:
                info = json_mod.loads(result.stdout)
            except json_mod.JSONDecodeError:
                return jsonify({
                    'success': False, 
                    'error': 'Failed to parse stream information. The camera may be sending an incomplete or corrupted stream.'
                }), 400
            
            # Check if we got any stream data
            if not info or 'streams' not in info or len(info.get('streams', [])) == 0:
                return jsonify({
                    'success': False, 
                    'error': 'No stream data received. The camera may need more time to send keyframes, or the stream path may be incorrect.'
                }), 400
            
            video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
            audio_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
            format_info = info.get('format', {})
            
            if not video_stream:
                return jsonify({
                    'success': False, 
                    'error': 'No video stream found in the response.'
                }), 400
            
            response_data = {
                'success': True,
                'raw': info,
                'video': video_stream,
                'audio': audio_stream,
                'format': format_info
            }
            
            if video_stream:
                response_data.update({
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'framerate': video_stream.get('r_frame_rate'),
                    'codec': video_stream.get('codec_name'),
                    'profile': video_stream.get('profile'),
                    'pix_fmt': video_stream.get('pix_fmt')
                })
                
            return jsonify(response_data)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/diagnostics/onvif', methods=['POST'])
    @login_required
    def diag_onvif():
        """Connect to an ONVIF camera and return detailed diagnostic info"""
        data = request.json
        host = data.get('host')
        port = int(data.get('port', 80))
        username = data.get('username')
        password = data.get('password')
        
        if not host or not username or not password:
            return jsonify({'success': False, 'error': 'Host, username, and password are required'}), 400
            
        prober = ONVIFProber()
        result = prober.get_detailed_diagnostics(host, port, username, password)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/diagnostics/ffmpeg-info')
    @login_required
    def diag_ffmpeg_info():
        """Get full FFmpeg version info"""
        try:
            ffmpeg_mgr = FFmpegManager()
            ffmpeg_exe = ffmpeg_mgr.get_ffmpeg_path()
            
            if not ffmpeg_exe:
                return jsonify({'success': False, 'error': 'FFmpeg not found'}), 404
            
            result = subprocess.run([ffmpeg_exe, '-version'], capture_output=True, text=True)
            return jsonify({
                'success': True,
                'version': result.stdout.split('\n')[0],
                'full_output': result.stdout
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/diagnostics/system-info')
    @login_required
    def diag_system_info():
        """Get detailed system information"""
        try:
            import platform
            
            # Fetch versions
            ffmpeg_mgr = FFmpegManager()
            ff_ver = ffmpeg_mgr.get_active_version()
            ff_ver_str = f"{ff_ver[0]}.{ff_ver[1]}.{ff_ver[2]}" if ff_ver else "Unknown"
            
            mm_ver = manager.mediamtx._get_latest_version()
            
            system_info = {
                'success': True,
                'platform': f"{platform.system()} {platform.release()} ({platform.machine()})",
                'python_version': sys.version.split()[0],
                'mediamtx_version': mm_ver,
                'ffmpeg_version': ff_ver_str
            }
            
            if psutil:
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                system_info.update({
                    'cpu_count': psutil.cpu_count(),
                    'total_memory': round(mem.total / (1024**3), 2),
                    'available_memory': round(mem.available / (1024**3), 2),
                    'disk_usage': disk.percent
                })
            else:
                system_info.update({
                    'cpu_count': 'Unknown',
                    'total_memory': 'Unknown',
                    'available_memory': 'Unknown',
                    'disk_usage': 0
                })
                
            return jsonify(system_info)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/server/restart', methods=['POST'])
    @login_required
    def server_restart():
        """Restart the application"""
        import threading
        def do_restart():
            import signal
            time.sleep(1)
            print("\n\nServer restart requested via UI...")
            manager.mediamtx.stop()
            for camera in manager.cameras:
                camera.stop()
            # Exit with code 42 to trigger restart (Linux) or just exit (Windows)
            if sys.platform.startswith('linux'):
                os._exit(42)
            else:
                os._exit(0)
        
        threading.Thread(target=do_restart, daemon=True).start()
        return jsonify({'success': True, 'message': 'Restarting...'})

    @app.route('/api/server/reboot', methods=['POST'])
    @login_required
    def server_reboot():
        """Reboot the host machine (Linux only)"""
        if not sys.platform.startswith('linux'):
            return jsonify({'success': False, 'error': 'Reboot is only supported on Linux'}), 400
            
        import threading
        def do_reboot():
            time.sleep(1)
            print("\n\nSystem reboot requested via UI...")
            manager.mediamtx.stop()
            # Send the command to reboot
            os.system('sudo reboot')
            
        threading.Thread(target=do_reboot, daemon=True).start()
        return jsonify({'success': True, 'message': 'Rebooting system...'})
    
    @app.route('/ip-management')
    @login_required
    def ip_management():
        whitelist = manager.get_ip_whitelist()
        return get_ip_management_html(whitelist)

    @app.route('/api/sessions', methods=['GET'])
    @login_required
    def get_sessions():
        return jsonify(manager.get_active_sessions())

    @app.route('/api/settings/whitelist', methods=['POST'])
    @login_required
    def save_whitelist():
        data = request.json
        whitelist = data.get('whitelist', [])
        try:
            manager.save_ip_whitelist(whitelist)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/auth', methods=['POST'])
    def mediamtx_auth():
        """Handle authentication requests from MediaMTX"""
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'Invalid request'}), 400
            
            client_ip = data.get('ip', '')
            
            # Normalize IPv6-mapped IPv4 addresses (e.g., ::ffff:127.0.0.1)
            if client_ip.startswith('::ffff:'):
                client_ip = client_ip.replace('::ffff:', '')
            
            user = data.get('user', '')
            password = data.get('password', '')
            
            if manager.debug_mode:
                 print(f"  [RTSP Auth Check] IP: {client_ip}, Path: {data.get('path')}, User: {user}")

            # 1. ALWAYS allow whitelisted IPs
            if manager.is_ip_whitelisted(client_ip):
                if manager.debug_mode:
                    print(f"  [RTSP] Auth bypass for whitelisted IP: {client_ip}")
                return '', 200
                
            # 2. Allow local loopback connections
            if client_ip in ['127.0.0.1', '127.0.1.1', '::1', 'localhost']:
                 if manager.debug_mode:
                     print(f"  [RTSP] Local bypass granted for {client_ip}")
                 return '', 200

            # 3. Check if global RTSP auth is enabled
            if not getattr(manager, 'rtsp_auth_enabled', False):
                return '', 200
                
            # 4. Validate credentials
            if user == manager.global_username and password == manager.global_password:
                return '', 200
                
            # print(f"  Auth: Denied for {user} from {client_ip}")
            return jsonify({'error': 'Unauthorized'}), 401
            
        except Exception as e:
            print(f"  Error in MediaMTX auth hook: {e}")
            return jsonify({'error': str(e)}), 500

    return app
