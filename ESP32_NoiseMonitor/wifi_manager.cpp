#include "wifi_manager.h"

WiFiManager::WiFiManager() {
  connected_ = false;
  lastReconnectAttempt_ = 0;
}

bool WiFiManager::begin() {
  Serial.println("\n📡 Connecting to WiFi...");
  Serial.printf("  SSID: %s\n", WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  unsigned long startAttemptTime = millis();
  
  while (WiFi.status() != WL_CONNECTED && 
         millis() - startAttemptTime < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    connected_ = true;
    Serial.println("\n✅ WiFi Connected!");
    Serial.printf("  IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Signal Strength: %d dBm\n", WiFi.RSSI());
    
    // Start UDP if using UDP mode
    #if !USE_HTTP_POST
    udp_.begin(SERVER_PORT);
    Serial.printf("  UDP initialized on port %d\n", SERVER_PORT);
    #endif
    
    return true;
  } else {
    Serial.println("\n❌ WiFi Connection Failed!");
    connected_ = false;
    return false;
  }
}

bool WiFiManager::isConnected() {
  return (WiFi.status() == WL_CONNECTED);
}

void WiFiManager::reconnect() {
  // Prevent too frequent reconnection attempts
  if (millis() - lastReconnectAttempt_ < 5000) {
    return;
  }
  
  lastReconnectAttempt_ = millis();
  
  Serial.println("🔄 Attempting WiFi reconnection...");
  WiFi.disconnect();
  delay(100);
  begin();
}

bool WiFiManager::sendUDP(const char* data) {
  if (!isConnected()) {
    return false;
  }
  
  IPAddress serverIP;
  if (!serverIP.fromString(SERVER_IP)) {
    Serial.println("❌ Invalid server IP address");
    return false;
  }
  
  udp_.beginPacket(serverIP, SERVER_PORT);
  udp_.write((const uint8_t*)data, strlen(data));
  bool success = udp_.endPacket();
  
  return success;
}

bool WiFiManager::sendHTTP(const char* data) {
  if (!isConnected()) {
    return false;
  }
  
  HTTPClient http;
  http.begin(HTTP_ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(data);
  
  bool success = (httpResponseCode == 200);
  
  if (!success) {
    Serial.printf("❌ HTTP POST failed: %d\n", httpResponseCode);
  }
  
  http.end();
  return success;
}

String WiFiManager::getLocalIP() {
  return WiFi.localIP().toString();
}

int WiFiManager::getRSSI() {
  return WiFi.RSSI();
}