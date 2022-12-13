# Multivariable Semester 1 Final Project
Kalman Filters (made by Steven Li and Eaton Huang)

For part of our first semester project, we decided to write an algorithm that can predict the location of an object with collected acceleration data.
Our project centers around things like torpedos, where GPS signal is unable to help with localization. The object is expected to travel at a constant velocity and an accelerometer will be used to constantly measure the acceleration of an object. This measurement will serve as an input to a Kalman Filter, which will reduce the noise and error. In the end, a predicted position along with a bivariate normal (3d Gaussian curve or nomal curve) will be generated.

To fully run this code you will need an Arduino UNO r3, WitMotion WT901, a usb-b to usb-a (or whatever your laptop needs), a serial converter, and a windowsOS (for calibration).
SETUP (calibrate WT901 if needed): 
1. Downlaod Arduino IDE (1.x version)
2. Install ArduSpreadsheet for the IDE (https://circuitjournal.com/arduino-serial-to-spreadsheet)
3. Build the following circuit and connect the arduino a laptop:
<img width="927" alt="Screenshot 2022-12-11 at 8 09 59 PM" src="https://user-images.githubusercontent.com/74156687/207230943-3ea457b0-d0be-4b17-8df2-d0849722856d.png">
4. Disconnect RX-TX pin and upload the code; after uploading, plug it back in and run ArduSpreadsheet
5. When the value in column A reaches 9.90, you can unplug the Arduino. Save the data as a csv to this folder.
6. If you want to plot the Actual values, measure the distance travelled in the x and y directions.
7. Open main.py; change the string in data_path to your filename and change the expected_vx value to distance in x/10 and change the expected_vy value to distance in y/10
8. Run the program!
