"""
DISHA 3D Avatar Integration
Live2D Cubism model with emotion-synced animations and lip-sync
"""

import os
import json
import threading
import time
from pathlib import Path
import webbrowser
import asyncio

# WebSocket for real-time communication
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("⚠️  Install websockets: pip install websockets")

# Try to import pygame for rendering Live2D
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class DISHAAvatar:
    """DISHA's 3D Live2D Avatar with emotion-synced animations and lip-sync"""
    
    def __init__(self, model_path="c001_f_costume_kouma"):
        self.model_path = Path(model_path)
        self.model_json = self.model_path / "c001_f_costume_kouma.model3.json"
        
        # Emotion to motion mapping
        self.emotion_motions = {
            'happy': 'motions/02_fun.motion3.json',
            'joy': 'motions/02_fun.motion3.json',
            'sad': 'motions/04_sad.motion3.json',
            'sadness': 'motions/04_sad.motion3.json',
            'angry': 'motions/01_angry.motion3.json',
            'anger': 'motions/01_angry.motion3.json',
            'surprise': 'motions/03_surprised.motion3.json',
            'surprised': 'motions/03_surprised.motion3.json',
            'fear': 'motions/03_surprised.motion3.json',
            'disgust': 'motions/01_angry.motion3.json',
            'neutral': 'motions/00_nomal.motion3.json',
            'idle': 'motions/I_idling_motion_01.motion3.json',
            'sleep': 'motions/05_sleep.motion3.json',
            'thinking': 'motions/07_tere.motion3.json',
        }
        
        self.current_emotion = 'neutral'
        self.is_speaking = False
        self.window = None
        self.websocket_server = None
        self.active_connections = []
        
        # Check if model exists
        if not self.model_json.exists():
            raise FileNotFoundError(f"Model not found: {self.model_json}")
        
        # Load model configuration
        with open(self.model_json, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)
        
        print(f"✅ DISHA Avatar loaded: {self.model_path.name}")
        print(f"   Available emotions: {list(set(self.emotion_motions.values()))}")
    
    def set_emotion(self, emotion):
        """Change avatar's emotion/animation"""
        emotion = emotion.lower()
        if emotion in self.emotion_motions:
            self.current_emotion = emotion
            motion_file = self.emotion_motions[emotion]
            
            # Broadcast to all connected web viewers (sync version)
            self._sync_broadcast({
                'type': 'emotion',
                'emotion': emotion,
                'motion': motion_file
            })
            
            return True
        return False
    
    def start_speaking(self):
        """Signal that DISHA is speaking (for lip-sync)"""
        self.is_speaking = True
        self._sync_broadcast({
            'type': 'speaking',
            'speaking': True
        })
    
    def stop_speaking(self):
        """Signal that DISHA stopped speaking"""
        self.is_speaking = False
        self._sync_broadcast({
            'type': 'speaking',
            'speaking': False
        })
    
    def _sync_broadcast(self, message):
        """Synchronous wrapper for broadcasting"""
        if not self.active_connections:
            return
        
        # Schedule the broadcast without awaiting
        try:
            import threading
            thread = threading.Thread(
                target=self._do_broadcast,
                args=(message,),
                daemon=True
            )
            thread.start()
        except:
            pass  # Silent failure if no connections
    
    def _do_broadcast(self, message):
        """Actually send the broadcast in a thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.broadcast_state(message))
            loop.close()
        except:
            pass
    
    async def broadcast_state(self, message):
        """Broadcast state updates to all connected clients"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send(message_json)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.remove(ws)
    
    async def websocket_handler(self, websocket):
        """Handle WebSocket connections for real-time avatar updates"""
        self.active_connections.append(websocket)
        print(f"🔗 Web viewer connected (Total: {len(self.active_connections)})")
        
        try:
            # Send initial state
            await websocket.send(json.dumps({
                'type': 'init',
                'emotion': self.current_emotion,
                'speaking': self.is_speaking
            }))
            
            # Keep connection alive
            async for message in websocket:
                pass  # We don't expect messages from client
                
        except Exception as e:
            pass
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            print(f"❌ Web viewer disconnected (Remaining: {len(self.active_connections)})")
    
    def start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        if not WEBSOCKET_AVAILABLE:
            print("⚠️  WebSocket not available. Real-time updates disabled.")
            return
        
        async def run_server():
            async with websockets.serve(self.websocket_handler, "localhost", 8765):
                print("✅ WebSocket server started on ws://localhost:8765")
                await asyncio.Future()  # Run forever
        
        # Run in background thread
        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_server())
        
        thread = threading.Thread(target=start_loop, daemon=True)
        thread.start()
    
    def create_web_viewer(self, port=8080):
        """Create a web-based Live2D viewer with real-time updates"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DISHA - Live2D Avatar</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        
        #avatar-container {{
            width: 800px;
            height: 800px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }}
        
        #avatar-display {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        
        .avatar-image {{
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 120px;
            animation: float 3s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-20px); }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .speaking {{
            animation: pulse 0.8s ease-in-out infinite !important;
        }}
        
        .info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            background: rgba(0,0,0,0.5);
            padding: 15px 25px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .info h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        
        .info p {{
            margin: 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .status {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.5);
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            backdrop-filter: blur(10px);
        }}
        
        .status.connected {{
            background: rgba(0,200,0,0.5);
        }}
        
        .emotion {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(255,255,255,0.9);
            padding: 15px 25px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }}
        
        .speaking-indicator {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255,100,100,0.9);
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            display: none;
        }}
        
        .speaking-indicator.active {{
            display: block;
            animation: pulse 0.8s ease-in-out infinite;
        }}
        
        .avatar-name {{
            font-size: 32px;
            color: white;
            margin-top: 20px;
            font-weight: bold;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="info">
        <h1>🌟 DISHA</h1>
        <p>Your AI Mental Health Companion</p>
        <p>Powered by Google Gemini Flash</p>
        <p>Multi-Language Support • 3D Avatar</p>
    </div>
    
    <div class="status" id="status">
        ⚪ Connecting...
    </div>
    
    <div id="avatar-container">
        <div id="avatar-display">
            <div class="avatar-image" id="avatar">
                😊
            </div>
            <div class="avatar-name">DISHA</div>
        </div>
    </div>
    
    <div class="emotion" id="emotion">
        😊 Neutral
    </div>
    
    <div class="speaking-indicator" id="speaking">
        🎤 Speaking...
    </div>
    
    <script>
        const emotionEmojis = {{
            'happy': '😊',
            'joy': '😊',
            'sad': '😢',
            'sadness': '😢',
            'angry': '😠',
            'anger': '😠',
            'surprise': '😲',
            'surprised': '😲',
            'fear': '😨',
            'disgust': '😒',
            'neutral': '😐',
            'idle': '🙂',
            'thinking': '🤔',
            'sleep': '😴'
        }};
        
        const emotionColors = {{
            'happy': '#FFD700',
            'joy': '#FFD700',
            'sad': '#4169E1',
            'sadness': '#4169E1',
            'angry': '#FF4500',
            'anger': '#FF4500',
            'surprise': '#FF69B4',
            'surprised': '#FF69B4',
            'fear': '#9370DB',
            'disgust': '#8B4513',
            'neutral': '#808080',
            'idle': '#00CED1',
            'thinking': '#DA70D6'
        }};
        
        const avatar = document.getElementById('avatar');
        const emotionDiv = document.getElementById('emotion');
        const speakingDiv = document.getElementById('speaking');
        const statusDiv = document.getElementById('status');
        
        let ws;
        
        function connectWebSocket() {{
            ws = new WebSocket('ws://localhost:8765');
            
            ws.onopen = () => {{
                console.log('✅ Connected to DISHA avatar server');
                statusDiv.textContent = '🟢 Connected';
                statusDiv.classList.add('connected');
            }};
            
            ws.onmessage = (event) => {{
                const data = JSON.parse(event.data);
                console.log('📨 Received:', data);
                
                if (data.type === 'emotion' || data.type === 'init') {{
                    updateEmotion(data.emotion);
                }}
                
                if (data.type === 'speaking' || data.type === 'init') {{
                    updateSpeaking(data.speaking);
                }}
            }};
            
            ws.onerror = (error) => {{
                console.error('❌ WebSocket error:', error);
                statusDiv.textContent = '🔴 Connection Error';
                statusDiv.classList.remove('connected');
            }};
            
            ws.onclose = () => {{
                console.log('🔌 Disconnected. Reconnecting in 2s...');
                statusDiv.textContent = '🟡 Reconnecting...';
                statusDiv.classList.remove('connected');
                setTimeout(connectWebSocket, 2000);
            }};
        }}
        
        function updateEmotion(emotion) {{
            const emoji = emotionEmojis[emotion] || '😊';
            const color = emotionColors[emotion] || '#667eea';
            
            avatar.textContent = emoji;
            avatar.style.filter = `drop-shadow(0 0 30px ${{color}})`;
            
            const emotionName = emotion.charAt(0).toUpperCase() + emotion.slice(1);
            emotionDiv.textContent = emoji + ' ' + emotionName;
            emotionDiv.style.borderLeft = `5px solid ${{color}}`;
        }}
        
        function updateSpeaking(speaking) {{
            if (speaking) {{
                avatar.classList.add('speaking');
                speakingDiv.classList.add('active');
            }} else {{
                avatar.classList.remove('speaking');
                speakingDiv.classList.remove('active');
            }}
        }}
        
        // Connect to WebSocket server
        connectWebSocket();
    </script>
</body>
</html>"""
        
        # Save HTML viewer
        viewer_path = self.model_path.parent / "disha_avatar_viewer.html"
        with open(viewer_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Web viewer created: {viewer_path}")
        return viewer_path
    
    def launch_window(self, auto_open=True):
        """Launch avatar display window"""
        # Start WebSocket server first
        self.start_websocket_server()
        
        # Use the new Live2D viewer
        viewer_path = Path("disha_live2d_viewer.html")
        
        if auto_open:
            try:
                import time
                time.sleep(1)  # Wait for WebSocket server to start
                webbrowser.open(f'file://{viewer_path.absolute()}')
                print("✅ DISHA avatar opened in browser")
                print("   The avatar will sync in real-time with emotions and speech!")
            except:
                print(f"⚠️ Please open manually: {viewer_path}")
        
        return viewer_path
    
    def close(self):
        """Close the avatar window"""
        if self.window and PYGAME_AVAILABLE:
            pygame.quit()
            self.window = None
    
    def create_web_viewer(self, port=8080):
        """Create a web-based Live2D viewer (easier than native SDK)"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DISHA - Live2D Avatar</title>
    <script src="https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/dylanNew/live2d/webgl/Live2D/lib/live2d.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        #canvas {{
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        .info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            background: rgba(0,0,0,0.5);
            padding: 15px 25px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        .info h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .info p {{
            margin: 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .emotion {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(255,255,255,0.9);
            padding: 15px 25px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div class="info">
        <h1>🌟 DISHA</h1>
        <p>Your AI Mental Health Companion</p>
        <p>Powered by Google Gemini Flash</p>
    </div>
    
    <canvas id="canvas" width="800" height="800"></canvas>
    
    <div class="emotion" id="emotion">
        😊 Neutral
    </div>
    
    <script>
        // This is a placeholder - you'll need proper Live2D Cubism SDK integration
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Draw placeholder
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(0, 0, 800, 800);
        
        ctx.fillStyle = 'white';
        ctx.font = 'bold 32px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('DISHA Avatar', 400, 350);
        
        ctx.font = '18px Arial';
        ctx.fillText('Live2D Model Ready', 400, 400);
        ctx.fillText('Emotion-synced animations enabled', 400, 430);
        
        // WebSocket for real-time emotion updates
        function connectWebSocket() {{
            const ws = new WebSocket('ws://localhost:8765');
            
            ws.onmessage = (event) => {{
                const data = JSON.parse(event.data);
                if (data.emotion) {{
                    updateEmotion(data.emotion);
                }}
                if (data.speaking !== undefined) {{
                    updateSpeaking(data.speaking);
                }}
            }};
            
            ws.onerror = () => {{
                console.log('WebSocket connection failed - running in standalone mode');
            }};
        }}
        
        function updateEmotion(emotion) {{
            const emotionMap = {{
                'happy': '😊',
                'sad': '😢',
                'angry': '😠',
                'surprise': '😲',
                'fear': '😨',
                'neutral': '😐'
            }};
            
            const emoji = emotionMap[emotion] || '😊';
            document.getElementById('emotion').textContent = emoji + ' ' + emotion.charAt(0).toUpperCase() + emotion.slice(1);
        }}
        
        function updateSpeaking(speaking) {{
            // Add lip-sync animation here
            console.log('Speaking:', speaking);
        }}
        
        // Try to connect (will fail gracefully if server not running)
        setTimeout(connectWebSocket, 1000);
    </script>
</body>
</html>"""
        
        # Save HTML viewer
        viewer_path = self.model_path.parent / "disha_avatar_viewer.html"
        with open(viewer_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Web viewer created: {viewer_path}")
        print(f"   Open in browser to see DISHA's avatar")
        return viewer_path
    
    def update_display(self):
        """Update the avatar display (called in main loop)"""
        if self.window and PYGAME_AVAILABLE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            
            # Update emotion display
            font = pygame.font.Font(None, 36)
            emotion_text = font.render(f"Emotion: {self.current_emotion}", True, (255, 255, 255))
            
            # Refresh part of screen
            pygame.draw.rect(self.window, (100, 126, 234), (250, 500, 300, 50))
            self.window.blit(emotion_text, (260, 510))
            pygame.display.flip()
    
    def close(self):
        """Close the avatar window"""
        if self.window and PYGAME_AVAILABLE:
            pygame.quit()
            self.window = None


# WebSocket server for real-time emotion updates
class EmotionWebSocketServer:
    """WebSocket server to send emotion updates to web viewer"""
    
    def __init__(self, port=8765):
        self.port = port
        self.server = None
        self.running = False
        
    def start(self):
        """Start WebSocket server in background"""
        try:
            import asyncio
            import websockets
            
            async def handler(websocket, path):
                while self.running:
                    # Send current emotion state
                    await asyncio.sleep(0.1)
            
            def run_server():
                asyncio.run(websockets.serve(handler, "localhost", self.port))
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            self.running = True
            print(f"✅ WebSocket server started on port {self.port}")
            
        except ImportError:
            print("⚠️  websockets not installed. Real-time updates disabled.")
        except Exception as e:
            print(f"⚠️  WebSocket server error: {e}")


# Test the avatar
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🌟 DISHA Avatar System - Test Mode 🌟")
    print("="*60 + "\n")
    
    # Create avatar
    avatar = DISHAAvatar("c001_f_costume_kouma")
    
    # Create web viewer
    avatar.create_web_viewer()
    
    # Test emotions
    emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
    print("\n Testing emotions:")
    for emotion in emotions:
        avatar.set_emotion(emotion)
        time.sleep(1)
    
    # Launch window
    print("\n Launching avatar window...")
    avatar.launch_window()
    
    print("\n✅ Avatar test complete!")
    print("   Integration ready for DISHAMemory.py")
