#include <Wire.h>
#include <JY901.h>

void setup() {
  Serial.begin(9600);
  delay(10000); //10 seconds delay for time to start ArduSpreadsheet + reconnect RX-TX pins

  // Data collection: collect 10 seconds of acceleration data (in x direction) with a 100ms interval
  // Download the ArduSpreadsheet tool for arduino and use it to save the data as a csv
  
  for (int i = 0; i < 100; i++) {
    serialEvent();
    // the extra stuff (sin/cos) accounts for gravitational acceleration
    float ax = ((float)JY901.stcAcc.a[0] / 32768 * 16)+sin((float)JY901.stcAngle.Angle[1]/32768*180*1000/57296);
    float ay = ((float)JY901.stcAcc.a[1] / 32768 * 16)-sin((float)JY901.stcAngle.Angle[0]/32768*180*1000/57296);
    float t = i * .1;
    Serial.print(t);
    Serial.print(",");
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.println();
    delay(100);
  }
}

void loop() {
  //only need data collection to run once
}

void serialEvent() {
  while (Serial.available()) {
    JY901.CopeSerialData(Serial.read());  //Call JY901 data cope function
  }
}