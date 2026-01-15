# Starting the ScoutSnout Backend Server

## Quick Start

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Start the server:**
   ```bash
   python3 app.py
   ```
   
   OR use the helper script:
   ```bash
   ./start_server.sh
   ```

3. **Verify it's running:**
   ```bash
   python3 check_server.py
   ```

## Important: IP Address Configuration

The Flutter app is currently configured to connect to: `http://192.168.8.169:5001`

**Your current local IP is:** `192.168.0.28`

### If your IP has changed, you need to update it in these Flutter files:

1. `lib/Components/snout_scanner.dart` (line ~125)
2. `lib/Pages/create_my_pet.dart` (line ~247)
3. `lib/Pages/scan_lost_pet.dart` (line ~15)
4. `lib/Components/lost_pet_scanner.dart` (line ~100)

Replace `192.168.8.169` with your current IP address (`192.168.0.28`).

### To find your current IP address:

**On macOS:**
```bash
ipconfig getifaddr en0  # For Wi-Fi
# or
ipconfig getifaddr en1  # For Ethernet
```

**Alternative method:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

## Server Configuration

The server runs on:
- **Host:** `0.0.0.0` (accessible from all network interfaces)
- **Port:** `5001`
- **Debug mode:** Enabled

This means it will be accessible at:
- Local: `http://localhost:5001`
- Network: `http://YOUR_IP_ADDRESS:5001`

## Troubleshooting

### Port already in use
If port 5001 is already in use:
```bash
lsof -ti:5001 | xargs kill -9
```

### Dependencies not installed
```bash
pip install -r requirements.txt
```

### Check if server is running
```bash
python3 check_server.py
```

### Test the server manually
```bash
curl http://localhost:5001/
```

## Network Requirements

For your iPhone to connect to the backend:
1. Your iPhone and Mac must be on the **same Wi-Fi network**
2. The Mac's firewall must allow connections on port 5001
3. The IP address in your Flutter app must match your Mac's current IP

## Notes

- The server runs in debug mode, so it will auto-reload on code changes
- Press `Ctrl+C` to stop the server
- Make sure to keep the terminal window open while developing












