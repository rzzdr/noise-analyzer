#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include "config.h"

class WiFiManager {
public:
  WiFiManager();
  
  bool begin();
  bool isConnected();
  void reconnect();
  
  // Data transmission methods
  bool sendUDP(const char* data);
  bool sendHTTP(const char* data);
  
  // Status methods
  String getLocalIP();
  int getRSSI();
  
private:
  WiFiUDP udp_;
  bool connected_;
  unsigned long lastReconnectAttempt_;
};

#endif // WIFI_MANAGER_H