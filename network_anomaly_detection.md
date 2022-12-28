##### Project Collaborators: *Andrew William, Dan Visitchaichan, Daniel Dinh, Michael Li*

### Project Details
This project required collaborative work with 4 other data scientists over 10 weeks where we defined the problem, explored and transformed our data, researched solutions, implemented machine learning models to solve the probelm outlined and presented our deliverables coordinators and other data science peers. The raw data was provided by CyAmast.

### Background
The Internet of Things (IoT) is a technology that connects various devices, machines, and systems to a single network. It is widely used in many industries because it allows for efficient data gathering. However, these devices often have limited functions and are vulnerable to cyber attacks such as DDoS and MITM, making it important to have protection measures in place, such as network anomaly detection.

### Data
The data provided contained time series network data including packet/byte counts in/out of a number of ports of a number of devices. Below is a snapshot:

 #   Column                                                             Dtype 
---  ------                                                             ----- 
 0   time                                                               object
 1   FromInternetTCPPort443IPurn:ietf:params:mud:controllerByteCount    int64 
 2   FromInternetTCPPort443IPurn:ietf:params:mud:controllerPacketCount  int64 
 3   FromInternetTCPPort554IPurn:ietf:params:mud:controllerByteCount    int64 
 4   FromInternetTCPPort554IPurn:ietf:params:mud:controllerPacketCount  int64 
 5   FromInternetTCPPort80IPurn:ietf:params:mud:controllerByteCount     int64 
 6   FromInternetTCPPort80IPurn:ietf:params:mud:controllerPacketCount   int64 
 7   FromInternetUDPPort67IPurn:ietf:params:mud:controllerByteCount     int64 
 8   FromInternetUDPPort67IPurn:ietf:params:mud:controllerPacketCount   int64 
 9   ToInternetRSVPPortAllIPurn:ietf:params:mud:controllerByteCount     int64 
 10  ToInternetRSVPPortAllIPurn:ietf:params:mud:controllerPacketCount   int64 
 11  ToInternetTCPPort443IPurn:ietf:params:mud:controllerByteCount      int64 
 12  ToInternetTCPPort443IPurn:ietf:params:mud:controllerPacketCount    int64 
 13  ToInternetTCPPort554IPurn:ietf:params:mud:controllerByteCount      int64 
 14  ToInternetTCPPort554IPurn:ietf:params:mud:controllerPacketCount    int64 
 15  ToInternetTCPPort80IPurn:ietf:params:mud:controllerByteCount       int64 
 16  ToInternetTCPPort80IPurn:ietf:params:mud:controllerPacketCount     int64 
 17  ToInternetUDPPort1024IPurn:ietf:params:mud:controllerByteCount     int64 
 18  ToInternetUDPPort1024IPurn:ietf:params:mud:controllerPacketCount   int64 
 19  ToLocalUDPPort5353IP224.0.0.251/32ByteCount                        int64 
 20  ToLocalUDPPort5353IP224.0.0.251/32PacketCount                      int64 
 21  ToLocalUDPPort67IP255.255.255.255/32ByteCount                      int64 
 22  ToLocalUDPPort67IP255.255.255.255/32PacketCount                    int64 
 23  device_mac                                                         object




