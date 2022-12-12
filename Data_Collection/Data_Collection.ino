#include <Wire.h>
#include <JY901.h>

void setup() {
  Serial.begin(9600);
  delay(10000);

  // Data collection: collect 10 seconds of acceleration data (in x direction) with a 100ms interval
  Serial.println("Data Collection Started");
  Serial.println("Time,Accel(x),Accel(y)");
  for (int i = 0; i < 100; i++) {
    serialEvent();
    float ax = ((float)JY901.stcAcc.a[0] / 32768 * 16);
    float ay = ((float)JY901.stcAcc.a[1] / 32768 * 16);
    float t = i * .1;
    Serial.print(t);
    Serial.print(",");
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.println();
    delay(100);
  }
  Serial.println("Data Collection Finished");
}

void loop() {
}

void serialEvent() {
  while (Serial.available()) {
    JY901.CopeSerialData(Serial.read());  //Call JY901 data cope function
  }
}
